# Phase 2 Review Notes (Person 2)

## Overall Verdict
The EDA and data-pipeline work is strong and ready for Phase 3/4 handoff.

## What Is Good
- Correct RSNA label extraction from detection CSV to image-level binary labels
- Clear class distribution analysis and visualization
- CLAHE demonstration is meaningful and aligned with medical-imaging practice
- Stratified 80/10/10 splitting is implemented
- Leakage checks are present and explicit
- DataLoader validation confirms expected tensor shapes and ranges

## Risks / Improvement Points
- `BASE_DIR` path in notebook is machine-specific and should be parameterized
- Ensure `data/processed/dataset_split.csv` is generated once and version-locked for all runs
- Keep train-only augmentations strictly out of validation/test
- Add a short written EDA conclusion block to summarize practical implications for training

## Phase 2 Sign-Off Conditions
- [x] EDA notebook covers structure, labels, class balance, and sample visuals
- [x] Split generation and leakage checks are implemented
- [x] Dataset class exists in `src/dataset.py`
- [x] Experiment protocol is documented in `experiment_protocol.md`
- [x] Tracking template is prepared in `src/experiment_tracking.py`

## Hand-Off To Phase 3/4
- Use only the frozen split from `data/processed/dataset_split.csv`
- Log Recall and F1 as top-priority metrics
- Keep model-comparison protocol identical across ConvNeXt and ViT
