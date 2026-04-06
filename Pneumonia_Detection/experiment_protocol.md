# Experiment Protocol - Pneumonia Detection (Phase 2)

## 1. Purpose
This protocol defines fixed rules for training and evaluating models in this project.
It is designed to keep comparisons fair and prevent data leakage or test-set tuning.

## 2. Task Definition
- Problem type: Binary classification
- Labels: 0 = Normal, 1 = Pneumonia
- Dataset: RSNA Pneumonia Detection Challenge only
- Non-clinical scope: Educational/research use only

## 3. Data Rules (Frozen)
- Split source: `data/processed/dataset_split.csv`
- Split policy: Stratified 80/10/10 train/val/test
- Leakage rule: Patient IDs must be disjoint across train/val/test
- No split regeneration during model experiments unless the whole experiment set is reset

## 4. Preprocessing Rules (Frozen)
- Read DICOM pixel arrays from raw data
- Apply CLAHE with:
  - `clipLimit=2.0`
  - `tileGridSize=(8, 8)`
- Resize to `224 x 224`
- Convert grayscale to 3 channels by channel replication
- Normalize with ImageNet mean/std:
  - mean `[0.485, 0.456, 0.406]`
  - std `[0.229, 0.224, 0.225]`

## 5. Augmentation Rules (Train Only)
- Random horizontal flip (p=0.5)
- Random rotation (+/-15 degrees)
- Brightness/contrast jitter (mild)
- Validation/test sets must use no random augmentation

## 6. Baseline and Improved Tracks
- Baseline model: ConvNeXt-Base (timm)
- Improved model: ViT-B/16 (HuggingFace)
- Fairness rule: both models must use the exact same split and evaluation protocol

## 7. Optimization and Stopping
- Loss: Cross-entropy with class weights (to address class imbalance)
- Scheduler: Cosine annealing
- Early stopping: monitor validation F1 and stop when no improvement for configured patience
- Best checkpoint: save by best validation F1

## 8. Metric Priority
Primary ranking order:
1. Recall (Sensitivity)
2. F1 Score
3. ROC-AUC
4. Precision
5. Confusion Matrix

## 9. Model Selection Rule
- Winner = best Recall + F1 on the frozen test set
- If ViT gain over ConvNeXt is marginal (<1%), present ConvNeXt as main model and ViT as controlled comparison

## 10. Test-Set Lock Rule
- Test set is for final evaluation only
- No hyperparameter tuning on test performance
- Final test metrics are reported once model choices are frozen

## 11. Tracking Requirements
Each run must log:
- Run name and timestamp
- Model family and checkpoint source
- Batch size, learning rate, weight decay, epochs, seed
- Train/val loss and metrics per epoch
- Final validation and test metrics
- Confusion matrix image and classification report
- Checkpoint artifact path

## 12. Reproducibility
- Fix random seed for Python, NumPy, and PyTorch
- Record environment details (Python, torch, CUDA, key package versions)
- Save command/config used to launch each run

## 13. Safety and Reporting
All public outputs must include:
"This model is for educational purposes only. It is not validated for clinical use."
