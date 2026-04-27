# config.py
# All hyperparameters and settings matching the paper exactly

import os

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_DIR = "E:/desktop/Cervical Cancer Project/Herlev Dataset original"
TRAIN_SUBDIR = "E:/desktop/Cervical Cancer Project/Herlev Dataset original/train"
TEST_SUBDIR  = "E:/desktop/Cervical Cancer Project/Herlev Dataset original/test"

# Sub-class → binary label mapping (paper: Normal=0, Cancerous=1)
# 7-Class labels
CLASS_MAP = {
    "carcinoma_in_situ": 0,
    "light_dysplastic": 1,
    "moderate_dysplastic": 2,
    "normal_columnar": 3,
    "normal_intermediate": 4,
    "normal_superficiel": 5,
    "severe_dysplastic": 6,
}

# 7 sub-class names (for confusion matrix labels etc.)
SUBCLASS_NAMES = [
    "carcinoma_in_situ",
    "light_dysplastic",
    "moderate_dysplastic",
    "normal_columnar",
    "normal_intermediate",
    "normal_superficiel",
    "severe_dysplastic",
]


# ── Pre-processing (CLAHE) ───────────────────────────────────────────────────
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_TILE_GRID    = (8, 8)

# ── Segmentation (AFKM) ─────────────────────────────────────────────────────
KMEANS_N_CLUSTERS  = 3          # foreground nucleus / cytoplasm / background
KMEANS_MAX_ITER    = 100
KMEANS_TOL         = 1e-4

AFKM_FUZZY_M       = 2          # fuzzy exponent (paper eq. 2)
AFKM_MAX_ITER      = 100
AFKM_TOL           = 1e-4
AFKM_N_CLUSTERS    = 3

# ── Feature Extraction ───────────────────────────────────────────────────────
N_FEATURES         = 40          # 20 nucleus + 20 cytoplasm (paper)
GLCM_DISTANCES     = [1]
GLCM_ANGLES        = [0, 0.785, 1.571, 2.356]   # 0°, 45°, 90°, 135°

# ── Classification ───────────────────────────────────────────────────────────
TRAIN_RATIO        = 0.70        # 70/30 split (paper)
CV_FOLDS           = 10          # 10-fold cross validation (paper)
RANDOM_STATE       = 42
N_CLASSES           = 7

# SVM kernels evaluated (paper Table 3)
SVM_KERNELS = {
    "linear":           {"kernel": "linear",  "C": 1.0},
    "2nd_poly":         {"kernel": "poly",    "C": 1.0, "degree": 2, "gamma": "scale"},
    "3rd_poly":         {"kernel": "poly",    "C": 1.0, "degree": 3, "gamma": "scale"},
    "RBF":              {"kernel": "rbf",     "C": 1.0, "gamma": "scale"},
}

# SVM_KERNELS = {
#     "RBF": {
#         "kernel": "rbf",
#         "C": 20,
#         "gamma": "scale"
#     }
# }

# MLP activation functions evaluated (paper Table 3)
MLP_CONFIGS = {
    "RBF":      {"hidden_layer_sizes": (80, 40), "activation": "relu",   "solver": "lbfgs",
                 "max_iter": 1000, "random_state": RANDOM_STATE},
    "Tanh":     {"hidden_layer_sizes": (80, 40), "activation": "tanh",   "solver": "lbfgs",
                 "max_iter": 1000, "random_state": RANDOM_STATE},
    "ReLu":     {"hidden_layer_sizes": (80, 40), "activation": "relu",   "solver": "lbfgs",
                 "max_iter": 1000, "random_state": RANDOM_STATE},
    "Sigmoid":  {"hidden_layer_sizes": (80, 40), "activation": "logistic","solver": "lbfgs",
                 "max_iter": 1000, "random_state": RANDOM_STATE},
}

# MLP_CONFIGS = {
#     "MLP": {
#         "hidden_layer_sizes": (256,128,64),
#         "activation": "relu",
#         "solver": "adam",
#         "max_iter": 600,
#         "early_stopping": True,
#         "random_state": RANDOM_STATE
#     }
# }

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR         = "outputs"
MODELS_DIR         = os.path.join(OUTPUT_DIR, "models")
FEATURES_CSV       = os.path.join(OUTPUT_DIR, "features.csv")
RESULTS_CSV        = os.path.join(OUTPUT_DIR, "table3_classification_results.csv")

FIG2_PATH          = os.path.join(OUTPUT_DIR, "fig2_clahe_preprocessing.png")
FIG3_PATH          = os.path.join(OUTPUT_DIR, "fig3_segmentation_comparison.png")
FIG5_PATH          = os.path.join(OUTPUT_DIR, "fig5_performance_bar_charts.png")
FIG6_PATH          = os.path.join(OUTPUT_DIR, "fig6_roc_auc_curves.png")

# ── Image ─────────────────────────────────────────────────────────────────────
IMG_SIZE           = (256, 256)   # resize for consistency
VALID_EXTENSIONS   = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
