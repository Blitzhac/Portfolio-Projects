import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pydicom
import cv2
from pathlib import Path
from torchvision import transforms


class RSNADataset(Dataset):
    """
    PyTorch Dataset for the RSNA Pneumonia Detection Challenge.
    Reads DICOM images, applies CLAHE preprocessing, and returns
    normalised 3-channel tensors ready for ConvNeXt-Base / ViT-B16.
    """

    def __init__(self, df, images_dir, transform=None):
        """
        df          : DataFrame with columns [patientId, label, split]
        images_dir  : Path to folder containing .dcm files
        transform   : Optional torchvision transforms (augmentations)
        """
        self.df         = df.reset_index(drop=True)   # clean integer index
        self.images_dir = Path(images_dir)
        self.transform  = transform

        # Keep only CLAHE params on the object. The actual OpenCV CLAHE object
        # is created lazily per process because it is not picklable on Windows.
        self._clahe_clip_limit = 2.0
        self._clahe_tile_grid_size = (8, 8)
        self._clahe = None

    def _get_clahe(self):
        """Create CLAHE lazily in each process to support DataLoader workers."""
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self._clahe_clip_limit,
                tileGridSize=self._clahe_tile_grid_size,
            )
        return self._clahe

    def __getstate__(self):
        """Drop non-picklable state when DataLoader spawns worker processes."""
        state = self.__dict__.copy()
        state["_clahe"] = None
        return state

    def __len__(self):
        """Returns total number of images in this split."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Loads, preprocesses and returns one image + its label.
        Called automatically by DataLoader for each item in a batch.
        """
        row       = self.df.iloc[idx]
        patient_id = row["patientId"]
        label     = int(row["label"])              # 0 = Normal, 1 = Pneumonia

        # ── 1. Read DICOM ──────────────────────────────────────────────────────
        dcm_path = self.images_dir / f"{patient_id}.dcm"
        pixels   = pydicom.dcmread(dcm_path).pixel_array   # shape: (H, W), uint8

        # ── 2. Apply CLAHE ─────────────────────────────────────────────────────
        pixels = self._get_clahe().apply(pixels)           # still (H, W), uint8

        # ── 3. Resize to 224×224 ───────────────────────────────────────────────
        pixels = cv2.resize(pixels, (224, 224),
                            interpolation=cv2.INTER_AREA)  # AREA best for downscaling

        # ── 4. Convert grayscale → 3 channels ─────────────────────────────────
        # ConvNeXt and ViT expect 3-channel input (pretrained on RGB ImageNet)
        # We copy the single channel 3 times: shape becomes (224, 224, 3)
        pixels = np.stack([pixels, pixels, pixels], axis=-1)

        # ── 5. Normalise to [0, 1] and convert to tensor ───────────────────────
        # Convert to float32 first, then scale to [0,1]
        pixels = pixels.astype(np.float32) / 255.0         # shape: (224, 224, 3)

        # PyTorch expects (C, H, W) not (H, W, C) — so we transpose
        pixels = torch.from_numpy(pixels).permute(2, 0, 1) # shape: (3, 224, 224)

        # ── 6. Apply augmentation transforms (training only) ───────────────────
        if self.transform:
            pixels = self.transform(pixels)                 # e.g. random flip, jitter

        return pixels, torch.tensor(label, dtype=torch.long)