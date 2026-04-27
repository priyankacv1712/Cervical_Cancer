# classification.py
# SVM and MLP classifiers with 10-fold CV, all 9 metrics, Figure 5 & 6, Table 3
# Exactly matches paper Section 3.2 and Table 3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from tqdm import tqdm

from sklearn.svm              import SVC
from sklearn.neural_network   import MLPClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import StratifiedKFold, cross_val_predict
from sklearn.metrics          import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, cohen_kappa_score,
    roc_curve, auc
)

# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_predict
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from config import (CV_FOLDS, RANDOM_STATE, SVM_KERNELS, MLP_CONFIGS,
                    MODELS_DIR, RESULTS_CSV, FIG5_PATH, FIG6_PATH, OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_score: np.ndarray = None) -> dict:
    """
    Compute the 9 performance indices used in the paper (Table 3):
      Sensitivity (Sen), Specificity (Spc), Precision (Pre),
      NPV, PPV, F1-score, MCC, Accuracy (Acc), Kappa score
    Plus AUC when y_score is supplied.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-10)          # recall / Sen
    specificity = tn / (tn + fp + 1e-10)          # Spc
    precision   = tp / (tp + fp + 1e-10)          # Pre / PPV
    npv         = tn / (tn + fn + 1e-10)          # Negative Predictive Value
    ppv         = precision
    f1          = f1_score(y_true, y_pred, zero_division=0)
    mcc         = matthews_corrcoef(y_true, y_pred)
    acc         = accuracy_score(y_true, y_pred)
    kappa       = cohen_kappa_score(y_true, y_pred)

    metrics = {
        "Sen":   round(sensitivity * 100, 1),
        "Spc":   round(specificity * 100, 1),
        "Pre":   round(precision   * 100, 1),
        "NPV":   round(npv         * 100, 1),
        "PPV":   round(ppv         * 100, 1),
        "F1":    round(f1          * 100, 1),
        "MCC":   round(mcc         * 100, 1),
        "Acc":   round(acc         * 100, 1),
        "Kappa": round(kappa       * 100, 1),
    }
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        metrics["AUC"] = round(auc(fpr, tpr), 3)
        metrics["fpr"] = fpr.tolist()
        metrics["tpr"] = tpr.tolist()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 10-fold CV evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_classifier(clf, X: np.ndarray, y: np.ndarray,
                         label: str, n_folds: int = CV_FOLDS) -> dict:
    """
    Run stratified 10-fold CV and return averaged metrics.
    """
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True,
                               random_state=RANDOM_STATE)
    y_pred_all  = np.zeros_like(y)
    y_score_all = np.zeros(len(y), dtype=np.float64)

    for train_idx, test_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        y_pred_all[test_idx] = clf.predict(X[test_idx])
        if hasattr(clf, "predict_proba"):
            y_score_all[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        elif hasattr(clf, "decision_function"):
            df = clf.decision_function(X[test_idx])
            # normalise to [0,1]
            y_score_all[test_idx] = (df - df.min()) / (df.ptp() + 1e-10)

    metrics = compute_all_metrics(y, y_pred_all, y_score_all)
    metrics["label"] = label
    return metrics

# def evaluate_classifier(clf, X, y, label, n_folds=10):
#     skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

#     y_pred = cross_val_predict(clf, X, y, cv=skf)

#     acc = accuracy_score(y, y_pred)

#     return {
#         "label": label,
#         "Acc": round(acc * 100, 2)
#     }

# ─────────────────────────────────────────────────────────────────────────────
# Build all classifiers
# ─────────────────────────────────────────────────────────────────────────────
def build_pipelines() -> dict:
    """
    Returns a dict of {display_name: sklearn Pipeline} for all
    SVM and MLP configurations from the paper.
    """
    pipelines = {}

    # SVM variants
    for name, kwargs in SVM_KERNELS.items():
        svm = SVC(probability=True, random_state=RANDOM_STATE,
                  class_weight="balanced", **kwargs)
        pipelines[f"SVM: {name}"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    svm),
        ])

    # MLP variants
    for name, kwargs in MLP_CONFIGS.items():
        mlp = MLPClassifier(**kwargs)
        pipelines[f"MLP: {name}"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    mlp),
        ])

    return pipelines

# def build_pipelines():
#     pipelines = {}

#     pipelines["SVM_RBF"] = Pipeline([
#         ("scaler", StandardScaler()),
#         ("clf", SVC(
#             kernel="rbf",
#             C=20,
#             gamma="scale",
#             probability=True,
#             class_weight="balanced",
#             random_state=42
#         ))
#     ])

#     pipelines["ExtraTrees"] = ExtraTreesClassifier(
#         n_estimators=500,
#         max_depth=None,
#         random_state=42
#     )

#     pipelines["RandomForest"] = RandomForestClassifier(
#         n_estimators=400,
#         random_state=42
#     )

#     pipelines["XGBoost"] = XGBClassifier(
#         n_estimators=500,
#         max_depth=8,
#         learning_rate=0.03,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         objective="multi:softprob",
#         eval_metric="mlogloss",
#         random_state=42
#     )

#     pipelines["MLP"] = Pipeline([
#         ("scaler", StandardScaler()),
#         ("clf", MLPClassifier(
#             hidden_layer_sizes=(256,128,64),
#             activation="relu",
#             max_iter=600,
#             early_stopping=True,
#             random_state=42
#         ))
#     ])

#     return pipelines


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Performance bar charts  (exact reproduction of paper Figure 5)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure5(results: list, save_path: str = FIG5_PATH) -> None:
    """
    Two grouped bar charts (one for SVM, one for MLP),
    one bar per classifier variant, one group per metric.
    Matches paper Figure 5 exactly.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metrics_order = ["Sen", "Spc", "Pre", "NPV", "PPV", "F1", "MCC", "Acc", "Kappa"]
    metric_labels = ["Sen.", "Spc.", "Pre.", "NPV", "PPV", "F1 Score", "MCC", "Acc.", "Kappa\nScore"]

    svm_results = [r for r in results if r["label"].startswith("SVM")]
    mlp_results = [r for r in results if r["label"].startswith("MLP")]

    def _svm_colour(label):
        c = {"SVM: linear": "#2196F3",
             "SVM: 2nd_poly": "#1565C0",
             "SVM: 3rd_poly": "#4CAF50",
             "SVM: RBF":      "#FF9800"}
        return c.get(label, "#888888")

    def _mlp_colour(label):
        c = {"MLP: Tanh":   "#FF8C00",
             "MLP: ReLu":   "#FFC04D",
             "MLP: Sigmoid": "#FFE099",
             "MLP: RBF":    "#FF6600"}
        return c.get(label, "#AAAAAA")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 12))

    def _plot_group(ax, subset, colour_fn, title_suffix):
        n_metrics  = len(metrics_order)
        n_clfs     = len(subset)
        bar_width  = 0.12
        x          = np.arange(n_metrics)

        offsets = np.linspace(-(n_clfs - 1) / 2, (n_clfs - 1) / 2, n_clfs) * bar_width

        for i, res in enumerate(subset):
            vals   = [res.get(m, 0) for m in metrics_order]
            colour = colour_fn(res["label"])
            ax.bar(x + offsets[i], vals, bar_width,
                   label=res["label"], color=colour, edgecolor="white", linewidth=0.5)

        ax.set_ylim(75, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylabel("Value in %", fontsize=11)
        ax.set_xlabel("Evaluation Metrics", fontsize=11)
        ax.legend(title="Classifier & Kernel\nFunction",
                  loc="lower right", fontsize=8, title_fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    _plot_group(ax1, svm_results, _svm_colour, "SVM")
    _plot_group(ax2, mlp_results, _mlp_colour, "MLP")

    plt.suptitle(
        "Figure 5. Performance analysis of SVM and MLP classifier\nfor various kernel functions",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure 5] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – ROC-AUC curves  (exact reproduction of paper Figure 6)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure6(results: list, save_path: str = FIG6_PATH) -> None:
    """
    Single ROC plot with all SVM and MLP variants.
    Matches paper Figure 6.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Colours matching paper Figure 6 legend
    colour_map = {
        "SVM: linear":   ("#1f77b4", "-",  "SVM: Linear"),
        "SVM: 2nd_poly": ("#ff7f0e", "-",  "SVM: 2nd Poly"),
        "SVM: 3rd_poly": ("#2ca02c", "-",  "SVM: 3rd Poly"),
        "SVM: RBF":      ("#d62728", "-",  "SVM: RBF"),
        "MLP: RBF":      ("#9467bd", "--", "MLP: RBF"),
        "MLP: Tanh":     ("#8c564b", "--", "MLP: Tan h"),
        "MLP: ReLu":     ("#e377c2", "--", "MLP: Relu"),
        "MLP: Sigmoid":  ("#7f7f7f", "--", "MLP: Sigmoid"),
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    for res in results:
        lbl   = res["label"]
        col, ls, display = colour_map.get(lbl, ("black", "-", lbl))
        fpr   = res.get("fpr", [0, 1])
        tpr   = res.get("tpr", [0, 1])
        auc_v = res.get("AUC", 0.0)
        ax.plot(fpr, tpr, color=col, linestyle=ls, linewidth=1.8,
                label=f"{display} (AUC={auc_v:.2f})", marker="o",
                markersize=3, markevery=20)

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)",      fontsize=11)
    ax.set_title("ROC Curves for SVM and MLP Classifiers", fontsize=12,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure 6] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 3 – save CSV and print
# ─────────────────────────────────────────────────────────────────────────────
def save_table3(results: list, save_path: str = RESULTS_CSV) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cols  = ["Classifier", "Sen.", "Spc.", "Pre.", "NPV", "PPV",
             "F1-score", "MCC", "Acc.", "Kappa score", "AUC"]
    rows  = []
    for r in results:
        rows.append({
            "Classifier":    r["label"],
            "Sen.":          r["Sen"],
            "Spc.":          r["Spc"],
            "Pre.":          r["Pre"],
            "NPV":           r["NPV"],
            "PPV":           r["PPV"],
            "F1-score":      r["F1"],
            "MCC":           r["MCC"],
            "Acc.":          r["Acc"],
            "Kappa score":   r["Kappa"],
            "AUC":           r.get("AUC", "N/A"),
        })

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(save_path, index=False)
    print(f"\n[Table 3] Saved → {save_path}")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Save best models
# ─────────────────────────────────────────────────────────────────────────────
def save_best_models(pipelines: dict, X: np.ndarray, y: np.ndarray) -> None:
    """
    Retrain and save the two best models (SVM 2nd_poly + MLP Tanh).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    best_svm_key = "SVM: 2nd_poly"
    best_mlp_key = "MLP: Tanh"

    for key, fname in [(best_svm_key, "svm_best.pkl"),
                       (best_mlp_key, "mlp_best.pkl")]:
        if key in pipelines:
            pipe = pipelines[key]
            pipe.fit(X, y)
            path = os.path.join(MODELS_DIR, fname)
            joblib.dump(pipe, path)
            print(f"[Model] Saved {key} → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_classification(X: np.ndarray, y: np.ndarray,
                       save_models: bool = True) -> list:
    """
    Train & evaluate all SVM / MLP classifiers.
    Returns list of result dicts.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipelines = build_pipelines()
    results   = []

    print(f"\nRunning 10-fold cross-validation on {len(X)} samples …")
    for name, pipe in tqdm(pipelines.items(), desc="Classifiers"):
        print(f"  ▶  {name}")
        res = evaluate_classifier(pipe, X, y, label=name)
        results.append(res)
        print(f"     Acc={res['Acc']}%  Sen={res['Sen']}%  "
              f"Spc={res['Spc']}%  AUC={res.get('AUC','N/A')}")

    # Generate all output figures / tables
    save_table3(results)
    generate_figure5(results)
    generate_figure6(results)

    if save_models:
        save_best_models(pipelines, X, y)
    # if save_models:
    #     best_result = max(results, key=lambda x: x["Acc"])
    #     best_name = best_result["label"]

    #     best_model = pipelines[best_name]
    #     best_model.fit(X, y)

    #     model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    #     joblib.dump(best_model, model_path)

    #     print(f"[Best Model Saved] {best_name} -> {model_path}")

    # return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from feature_extraction import build_feature_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="herlev dataset original")
    parser.add_argument("--features_csv", type=str, default=None,
                        help="Path to pre-computed features CSV (skip extraction)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.features_csv and os.path.isfile(args.features_csv):
        print(f"Loading pre-computed features from {args.features_csv}")
        df = pd.read_csv(args.features_csv)
        y  = df["label"].values.astype(np.int32)
        X  = df.drop(columns=["filepath", "label", "subclass"],
                     errors="ignore").values.astype(np.float64)
    else:
        from config import FEATURES_CSV
        X, y, _, _ = build_feature_dataset(
            args.data_dir, split="train", save_csv=FEATURES_CSV
        )

    run_classification(X, y)
