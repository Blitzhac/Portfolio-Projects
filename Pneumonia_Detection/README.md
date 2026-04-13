# Pneumonia Detection from Chest X-Rays

> **Disclaimer:** This is an educational and research project only. It is explicitly **not intended for clinical diagnosis** or any medical decision-making.

---

## Project Overview

A deep learning pipeline for binary classification of chest X-ray images (Normal vs Pneumonia), built from scratch using PyTorch. The project compares a CNN-based baseline (ConvNeXt-Base) against a Vision Transformer (ViT-B/16), with Grad-CAM explainability and a live Streamlit demo.

**Team:**
- Person 1 — Data pipeline, EDA, CNN baseline, evaluation
- Person 2 — ViT track, Grad-CAM, Streamlit app, portfolio

**Dataset:** [RSNA Pneumonia Detection Challenge — Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)

**Success metrics:** F1 Score · Recall (Sensitivity) · Precision · ROC-AUC · Confusion Matrix

---

## Development Environment

This project uses two environments depending on the task:

| Environment | GPU | Used For |
|---|---|---|
| **Google Colab (free)** | T4 — 15GB VRAM | Phase 3 (CNN training), Phase 4 (ViT training), Phase 5 (Final evaluation) |
| **Jupyter Notebook (local)** | Local CPU / 6GB GPU | Phase 1 (Project brief), Phase 2 (EDA), Phase 6 (Streamlit app) |

> **Why Colab for training?** The free T4 GPU gives 15GB VRAM — enough to run ConvNeXt-Base and ViT-B/16 comfortably. Always save checkpoints to Google Drive during training so you don't lose progress if the session disconnects.

---

## Roadmap

### Phase 1 — Scope Lock (Day 1–2)
**Owner: Both**

- Define objective: binary classification — Normal vs Pneumonia
- Lock success metrics: F1, Recall, Precision, ROC-AUC, confusion matrix
- Metric priority: Recall first (false negatives = missed pneumonia = dangerous)
- Freeze scope to one dataset only — RSNA Pneumonia Detection Challenge (radiologist-verified, 30,000 images)
- Establish communication rule: educational/research project, not for clinical diagnosis
- Set up shared GitHub repo with branch strategy (`p1/data-pipeline`, `p2/experiments`)

**Deliverables:**
- `project_brief.md`
- `requirements.txt`
- README skeleton with disclaimer

---

### Phase 2 — Data Pipeline & EDA (Week 1)
**Owner: Person 1 (Person 2 reviews + sets up experiment tracking)**
**Environment: Jupyter Notebook (local)**

**Concept — Why CLAHE?**
Standard normalisation treats all pixels equally. CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast in X-rays, making subtle opacities in lung tissue visible to the model. It is standard practice in medical imaging pipelines.

**Note on the RSNA dataset structure:**
The RSNA dataset includes bounding box annotations (designed for object detection), but we only need image-level labels for binary classification. The dataset provides a `stage_2_train_labels.csv` file — any image with at least one bounding box entry is labelled Pneumonia; images with no entry are Normal. This extraction step happens first, before any split or preprocessing.

**Person 1 tasks:**
- Download RSNA dataset via Kaggle CLI (`kaggle competitions download -c rsna-pneumonia-detection-challenge`)
- Extract image-level labels from `stage_2_train_labels.csv` (Pneumonia vs Normal)
- Explore dataset: class distribution, image sizes, pixel intensity stats
- Write EDA notebook with sample images and class balance observations
- Build train/val/test split (80/10/10, stratified) from scratch
- Implement CLAHE preprocessing pipeline
- Build PyTorch `Dataset` and `DataLoader` classes
- Apply medical-safe augmentations: random horizontal flip, rotation ±15°, brightness/contrast jitter
- Verify no data leakage across splits

**Person 2 tasks:**
- Review EDA notebook and add observations
- Research MixUp augmentation — prepare to implement in Phase 3
- Set up experiment tracking (Weights & Biases or MLflow)
- Write and finalise experiment protocol document

**Deliverables:**
- `notebooks/eda.ipynb`
- `src/dataset.py`
- `experiment_protocol.md`
- Class distribution plot

---

### Phase 3 — CNN Baseline: ConvNeXt-Base (Week 2)
**Owner: Person 1 (Person 2 implements Grad-CAM in parallel)**
**Environment: Google Colab (T4 GPU)**

**Concept — Why ConvNeXt-Base?**
ConvNeXt (Meta AI, 2022) is a CNN redesigned using everything learned from Vision Transformers — modern layer normalisation, large kernels, depthwise convolutions, and inverted bottlenecks. ConvNeXt-Base achieves 85.8% on ImageNet and fits comfortably on Colab's T4 GPU (15GB VRAM). It is a stronger and more modern baseline than EfficientNetV2-B3, and makes the Phase 4 CNN vs ViT comparison more meaningful. Training uses cosine annealing LR scheduling and checkpoint saving to Google Drive to handle Colab session limits.

**Person 1 tasks:**
- Mount Google Drive and set up checkpoint saving
- Load ConvNeXt-Base via `timm`, freeze backbone, train classification head (5 epochs)
- Unfreeze last 2 blocks and fine-tune (10–15 epochs)
- Implement cosine annealing LR scheduler + early stopping
- Implement class-weighted loss to handle the ~1:3 class imbalance
- Log all metrics to W&B / MLflow
- Plot confusion matrix and full classification report

**Person 2 tasks:**
- Implement Grad-CAM for the baseline model
- Generate heatmap overlays on 20 sample images
- Review training logs and flag overfitting or instability
- Write baseline results observation notes

**Deliverables:**
- `src/train_cnn.py`
- `results/baseline_results.csv`
- `results/gradcam_samples/`
- `results/confusion_matrix.png`

---

### Phase 4 — ViT Improved Track: ViT-B/16 (Week 3)
**Owner: Person 2 (Person 1 reviews for fair comparison)**
**Environment: Google Colab (T4 GPU)**

**Concept — Why ViT-B/16?**
Vision Transformers use self-attention across image patches rather than local convolution filters. This allows the model to capture long-range dependencies across the full X-ray — for example, relating opacity patterns in the left lung to the right. We use ViT-B/16 pretrained on ImageNet-21k via HuggingFace Transformers, which provides stronger pretraining than standard ImageNet. The same dataset split, augmentations, and evaluation protocol from Phase 3 must be used to ensure a fair comparison.

**Person 2 tasks:**
- Load ViT-B/16 from HuggingFace, adapt classification head
- Use identical DataLoader from Phase 2 — same split, no exceptions
- Fine-tune with same training protocol as CNN baseline
- Log all metrics with same W&B / MLflow config
- Generate Attention Rollout visualisations (ViT equivalent of Grad-CAM)
- Build side-by-side comparison table: CNN vs ViT

**Person 1 tasks:**
- Review ViT training logs and verify identical split was used
- Run leakage check — confirm no test-set tuning occurred
- Jointly decide with Person 2 whether a lightweight ensemble is worth attempting

**Deliverables:**
- `src/train_vit.py`
- `results/vit_results.csv`
- `results/attention_rollout/`
- `results/cnn_vs_vit_comparison.md`

---

### Phase 5 — Final Evaluation & Error Analysis (Week 4)
**Owner: Both**
**Environment: Google Colab (T4 GPU)**

**Concept — Why Test-Time Augmentation (TTA)?**
TTA runs inference multiple times on slightly different versions of each test image (flipped, rotated) and averages the predictions. This costs no additional training and typically adds 1–2% to test accuracy — a free boost for your final reported scores.

**Person 1 tasks:**
- Run one-time final test evaluation on CNN best checkpoint
- Implement TTA on the test set and compare with/without
- Analyse false negatives: pull misclassified images and inspect Grad-CAM overlays
- Write limitations section covering dataset bias and domain shift risk

**Person 2 tasks:**
- Run one-time final test evaluation on ViT best checkpoint
- Write model selection rationale using the decision rule: best Recall + F1 wins; if ViT gain is marginal, use CNN as main story and present ViT as a controlled comparison
- Draft the final results report with all metric panels
- Write safety disclaimer section for all public-facing outputs

**Deliverables:**
- `results/final_results_report.md`
- `results/false_negative_analysis/`
- `results/tta_comparison.csv`

---

### Phase 6 — Streamlit Demo & Portfolio Packaging (Week 4–5)
**Owner: Person 2 (Streamlit app) + Person 1 (GitHub & Kaggle)**
**Environment: Jupyter Notebook (local)**

**Concept — Why a Streamlit demo?**
A live demo where anyone can upload an X-ray and see the prediction alongside a Grad-CAM heatmap overlay is far more memorable to recruiters than a static notebook. It shows full-stack ML skills: model, inference pipeline, and deployment. Deployed for free on HuggingFace Spaces.

**Person 2 tasks:**
- Build Streamlit app: upload image → preprocess → predict → overlay Grad-CAM heatmap
- Export best model to ONNX for fast CPU inference in the app
- Deploy to HuggingFace Spaces (free tier)
- Write HuggingFace model card with intended use, metrics, and safety limitations

**Person 1 tasks:**
- Polish GitHub repo: README, badges, usage instructions, reproducibility steps
- Publish Kaggle notebook: EDA + training story narrative
- Write resume bullets from measured outcomes only — no inflated claims
- Draft LinkedIn post with one key insight from error analysis

**Deliverables:**
- `app.py` (Streamlit)
- `models/model.onnx`
- HuggingFace Space (public link)
- GitHub README (polished)
- Kaggle public notebook

---

## Dependency Order

```
Phase 1 → Phase 2 → Phase 3 ──────────────────────────→ Phase 5
                         └──→ Phase 4 (parallel) ──────→ Phase 5
                         └──→ Phase 6 (can start after Phase 3 is stable)
                                           └──→ Phase 7 (resume/LinkedIn)
```

- Phases 1–2 must complete before any model work
- Phase 3 blocks Phase 4 (ViT requires baseline as reference)
- Phase 5 requires both Phase 3 and Phase 4 to be frozen
- Phase 6 can begin as soon as Phase 3 is stable
- Grad-CAM (Person 2) runs in parallel with CNN training (Person 1) in Phase 3

---

## Verification Checklist

- [ ] Scope confirmed: binary classification, RSNA dataset only, no scope creep
- [ ] Experiment protocol written before any model training begins
- [ ] CLAHE preprocessing applied consistently across all splits
- [ ] No data leakage: train/val/test splits are non-overlapping
- [ ] CNN and ViT use identical split, augmentations, and evaluation protocol
- [ ] Final test set evaluated exactly once, after all model choices are frozen
- [ ] Final report includes Recall, F1, ROC-AUC — not accuracy alone
- [ ] Error analysis focuses on false negatives
- [ ] All public text includes non-clinical-use disclaimer
- [ ] Three visible portfolio artifacts: GitHub repo, Kaggle notebook, HuggingFace Space

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch + torchvision |
| CNN baseline | ConvNeXt-Base (via `timm`) |
| Transformers | HuggingFace `transformers` (ViT-B/16) |
| Image processing | OpenCV, Pillow, pydicom (RSNA uses DICOM format) |
| Experiment tracking | Weights & Biases or MLflow |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| Training environment | Google Colab (T4 GPU — free) |
| EDA environment | Jupyter Notebook (local) |
| Demo app | Streamlit |
| Deployment | HuggingFace Spaces |
| Version control | Git + GitHub |
| Dataset | RSNA Pneumonia Detection Challenge (Kaggle) |

---

## Project Structure

```
pneumonia-detection/
├── data/
│   ├── raw/               ← original Kaggle dataset (not pushed to GitHub)
│   └── processed/         ← CLAHE-processed images
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── dataset.py
│   ├── train_cnn.py
│   ├── train_vit.py
│   └── gradcam.py
├── models/                ← saved checkpoints (not pushed to GitHub)
├── results/
│   ├── baseline_results.csv
│   ├── vit_results.csv
│   └── final_results_report.md
├── app.py                 ← Streamlit demo
├── requirements.txt
├── experiment_protocol.md
└── README.md
```

---

*Built for learning purposes. Always consult a qualified medical professional for any health concerns.*