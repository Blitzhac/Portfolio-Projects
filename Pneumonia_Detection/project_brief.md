# Project Brief — Pneumonia Detection from Chest X-Rays

## Objective
Binary classification of chest X-ray images into two categories:
- Normal (healthy lungs)
- Pneumonia (infected lungs)

This is an educational and research project.
It is explicitly NOT intended for clinical diagnosis.

## Dataset
- Name: RSNA Pneumonia Detection Challenge
- Source: Kaggle (kaggle competitions download -c rsna-pneumonia-detection-challenge)
- Size: ~30,000 chest X-ray images
- Format: DICOM (.dcm)
- Labels: Radiologist-verified binary labels (Normal / Pneumonia)

## Scope (frozen — no changes after this document is signed off)
- Binary classification only — Normal vs Pneumonia
- Single dataset only — RSNA, no mixing with other sources
- No multi-class disease detection
- No clinical-grade claims in any outputs

## Success Metrics (in priority order)
1. Recall (Sensitivity) — primary metric, must be maximised
2. F1 Score
3. ROC-AUC
4. Precision
5. Confusion Matrix

## Model Selection Rule
Choose the model with the best Recall + F1 on the test set.
If ViT gain over ConvNeXt-Base is marginal (<1%), use ConvNeXt-Base
as the main story and present ViT as a controlled comparison.

## Models
- Phase 3 baseline : ConvNeXt-Base (pretrained ImageNet, via timm)
- Phase 4 improved : ViT-B/16 (pretrained ImageNet-21k, via HuggingFace)

## Environments
- Training (Phase 3, 4, 5) : Google Colab — T4 GPU (15GB VRAM)
- EDA + App (Phase 2, 6)   : Jupyter Notebook — local machine

## Team
- Person 1 : Data pipeline, EDA, CNN training, final evaluation
- Person 2 : ViT training, Grad-CAM, Streamlit app, portfolio

## Known Risks
- Class imbalance in RSNA dataset — handled via class-weighted loss
- Colab session disconnects — handled via Google Drive checkpointing
- Domain shift — model trained on RSNA may not generalise to
  other hospitals (documented in limitations section)
- Dataset bias — RSNA images sourced from specific demographics
  (documented in limitations section)

## Safety
All public outputs (GitHub, Kaggle, HuggingFace) must include:
"This model is for educational purposes only.
 It is not validated for clinical use."

## Sign-off
- Person 1: _______________  Date: _______________
- Person 2: _______________  Date: _______________