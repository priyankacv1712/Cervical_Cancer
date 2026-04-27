# Automated Real-Time Cervical Cancer Diagnosis

Implementation of the paper:
> **"Automated real-time cervical cancer diagnosis using NVIDIA Jetson Nano"**  
> Pallavi Mulmule et al., *Bulletin of Electrical Engineering and Informatics*, Vol. 14, No. 5, October 2025

---

## Overview

This project implements an automated framework for cervical cell analysis using:
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for pre-processing
- **AFKM** (Adaptive Fuzzy K-Means) segmentation
- **40 handcrafted features** from nucleus and cytoplasm (morphological, textural, intensity-based)
- **SVM** (2nd order polynomial kernel) — best accuracy: **96%**
- **MLP** (Tanh activation) — best accuracy: **97%**

---

## Dataset Structure

```
herlev dataset original/
├── train/
│   ├── carcinoma_in_situ/
│   ├── light_dysplastic/
│   ├── moderate_dysplastic/
│   ├── normal_columnar/
│   ├── normal_intermediate/
│   ├── normal_superficiel/
│   └── severe_dysplastic/
└── test/
    ├── carcinoma_in_situ/
    ├── light_dysplastic/
    ├── moderate_dysplastic/
    ├── normal_columnar/
    ├── normal_intermediate/
    ├── normal_superficiel/
    └── severe_dysplastic/
```

The dataset contains **917 single-cell Pap smear images** from the Herlev dataset.

Class mapping (as per paper):
- **Normal**: normal_columnar, normal_intermediate, normal_superficiel
- **Cancerous**: carcinoma_in_situ, light_dysplastic, moderate_dysplastic, severe_dysplastic

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Full Pipeline (Pre-processing → Segmentation → Feature Extraction → Classification)

```bash
python src/main.py --data_dir "herlev dataset original"
```

### 2. Step-by-step

```bash
# Pre-processing + Segmentation visualization (Figure 2 & 3)
python src/preprocessing.py --data_dir "herlev dataset original"

# Feature extraction
python src/feature_extraction.py --data_dir "herlev dataset original"

# Train & evaluate classifiers (Table 3, Figure 5, Figure 6)
python src/classification.py --data_dir "herlev dataset original"
```

### 3. All outputs at once

```bash
python src/main.py --data_dir "herlev dataset original" --save_all
```

---

## Outputs

All outputs are saved in the `outputs/` folder:

| File | Description |
|------|-------------|
| `fig2_clahe_preprocessing.png` | CLAHE pre-processing result (Figure 2) |
| `fig3_segmentation_comparison.png` | K-Means vs AFKM segmentation (Figure 3) |
| `fig5_performance_bar_charts.png` | Performance bar charts for all classifiers (Figure 5) |
| `fig6_roc_auc_curves.png` | ROC-AUC curves for all classifiers (Figure 6) |
| `table3_classification_results.csv` | Full classification metrics (Table 3) |
| `models/svm_best.pkl` | Saved best SVM model |
| `models/mlp_best.pkl` | Saved best MLP model |
| `features.csv` | Extracted 40-feature dataset |

---

## Results (from paper)

| Classifier | Accuracy | Sensitivity | Specificity | AUC |
|-----------|----------|-------------|-------------|-----|
| SVM (2nd order poly) | **96%** | 96% | 99% | 0.97 |
| MLP (Tanh) | **97%** | 97% | 92% | 0.94 |

---

## Architecture

```
Input Image
    ↓
Pre-processing (CLAHE)
    ↓
Segmentation (AFKM)
    ↓
Feature Extraction (40 features: nucleus + cytoplasm)
    ↓
Classifier Training (SVM / MLP) with 10-fold cross-validation
    ↓
Prediction (Normal / Cancerous)
```

---

## Citation

```bibtex
@article{mulmule2025cervical,
  title={Automated real-time cervical cancer diagnosis using NVIDIA Jetson Nano},
  author={Mulmule, Pallavi and Shilaskar, Swati and Bhatlawande, Shripad and Mulmule, Vedant and Kamble, Vaishali H and Madake, Jyoti},
  journal={Bulletin of Electrical Engineering and Informatics},
  volume={14},
  number={5},
  pages={4074--4085},
  year={2025},
  doi={10.11591/eei.v14i5.10169}
}
```
