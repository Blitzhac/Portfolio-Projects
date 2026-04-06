"""Lightweight tracking helpers for W&B-based experiment logging.

This module keeps a single logging interface so both CNN and ViT scripts can
share the same tracking structure and metric names.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

try:
    wandb = import_module("wandb")
except Exception:  # pragma: no cover
    wandb = None


@dataclass
class RunConfig:
    project: str
    run_name: str
    model_name: str
    split_csv: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    notes: str = ""
    tags: Optional[list[str]] = None


class ExperimentTracker:
    """Wrapper around Weights & Biases with safe no-op fallback."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and (wandb is not None)
        self._run = None

    def init(self, cfg: RunConfig) -> None:
        if not self.enabled:
            print("[tracking] W&B disabled or unavailable. Running without remote tracking.")
            return

        self._run = wandb.init(
            project=cfg.project,
            name=cfg.run_name,
            config=asdict(cfg),
            tags=cfg.tags or [],
            notes=cfg.notes,
        )

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_precision: float,
        val_recall: float,
        val_f1: float,
        val_roc_auc: float,
        learning_rate: float,
    ) -> None:
        payload = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/precision": val_precision,
            "val/recall": val_recall,
            "val/f1": val_f1,
            "val/roc_auc": val_roc_auc,
            "optimizer/lr": learning_rate,
        }
        self._log(payload)

    def log_final_test(
        self,
        test_precision: float,
        test_recall: float,
        test_f1: float,
        test_roc_auc: float,
    ) -> None:
        payload = {
            "test/precision": test_precision,
            "test/recall": test_recall,
            "test/f1": test_f1,
            "test/roc_auc": test_roc_auc,
        }
        self._log(payload)

    def log_artifact(self, file_path: str, artifact_name: str, artifact_type: str = "model") -> None:
        if not self.enabled:
            return

        p = Path(file_path)
        if not p.exists():
            print(f"[tracking] Artifact path does not exist: {p}")
            return

        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_file(str(p))
        wandb.log_artifact(artifact)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            wandb.finish()

    def _log(self, payload: Dict[str, Any]) -> None:
        if self.enabled:
            wandb.log(payload)
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[tracking:{timestamp}] {payload}")


if __name__ == "__main__":
    # Minimal example usage for quick validation.
    tracker = ExperimentTracker(enabled=True)
    cfg = RunConfig(
        project="pneumonia-detection",
        run_name="vit_phase4_dryrun",
        model_name="vit-base-patch16-224",
        split_csv="data/processed/dataset_split.csv",
        seed=42,
        epochs=1,
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=1e-2,
        notes="tracking module smoke test",
        tags=["phase2", "tracking-template"],
    )
    tracker.init(cfg)
    tracker.log_epoch(
        epoch=1,
        train_loss=0.71,
        val_loss=0.66,
        val_precision=0.81,
        val_recall=0.84,
        val_f1=0.82,
        val_roc_auc=0.89,
        learning_rate=1e-4,
    )
    tracker.log_final_test(
        test_precision=0.80,
        test_recall=0.85,
        test_f1=0.82,
        test_roc_auc=0.90,
    )
    tracker.finish()
