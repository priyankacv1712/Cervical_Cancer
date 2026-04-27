# main.py
# Full pipeline: preprocessing → segmentation → feature extraction → classification
# Reproduces all paper outputs (Figure 2, 3, 5, 6, Table 3)

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import OUTPUT_DIR, FEATURES_CSV, MODELS_DIR
from preprocessing    import collect_image_paths, generate_figure2
from segmentation     import generate_figure3
from feature_extraction import build_feature_dataset
from classification   import run_classification
from config import SUBCLASS_NAMES



# ─────────────────────────────────────────────────────────────────────────────
def pick_sample_images(data_dir: str, split: str, n: int = 4) -> list:
    """Pick one image per sub-class (up to n) for visualisation figures."""
    paths      = collect_image_paths(data_dir, split)
    sample     = []
    seen_dirs  = set()
    for p in paths:
        d = os.path.basename(os.path.dirname(p))
        if d not in seen_dirs:
            sample.append(p)
            seen_dirs.add(d)
        if len(sample) == n:
            break
    return sample


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Automated Cervical Cancer Detection (Mulmule et al., 2025)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="herlev dataset original",
        help="Root directory containing train/ and test/ sub-folders"
    )
    parser.add_argument(
        "--save_all", action="store_true",
        help="Generate all paper figures (Figs 2, 3, 5, 6) + Table 3"
    )
    parser.add_argument(
        "--skip_extraction", action="store_true",
        help="Skip feature extraction and load from existing features.csv"
    )
    parser.add_argument(
        "--features_csv", type=str, default=None,
        help="Path to pre-computed features CSV (overrides default)"
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 65)
    print(" Automated Real-Time Cervical Cancer Diagnosis")
    print(" Mulmule et al. – BEEI Vol.14, No.5, Oct 2025")
    print("=" * 65)
    print(f" Dataset : {args.data_dir}")
    print(f" Outputs : {OUTPUT_DIR}/")
    print("=" * 65)

    # ── Step 1: Visual figures (Figure 2 & 3) ────────────────────────────
    if args.save_all:
        print("\n[Step 1] Generating Figure 2 (CLAHE pre-processing) …")
        samples = pick_sample_images(args.data_dir, "train", n=4)
        if samples:
            generate_figure2(samples)
        else:
            print("  [WARN] No training images found for Figure 2.")

        print("\n[Step 2] Generating Figure 3 (Segmentation comparison) …")
        if samples:
            generate_figure3(samples)
        else:
            print("  [WARN] No training images found for Figure 3.")
    else:
        print("\n[Step 1-2] Skipping figure generation (use --save_all to enable)")

    # ── Step 2: Feature extraction ────────────────────────────────────────
    feat_csv = args.features_csv or FEATURES_CSV
    if args.skip_extraction and os.path.isfile(feat_csv):
        print(f"\n[Step 3] Loading pre-computed features from {feat_csv} …")
        import pandas as pd
        df = pd.read_csv(feat_csv)
        y  = df["label"].values.astype(np.int32)
        X  = df.drop(columns=["filepath", "label", "subclass"],
                     errors="ignore").values.astype(np.float64)
        print(f"  Loaded: {X.shape[0]} samples × {X.shape[1]} features")
    else:
        print("\n[Step 3] Extracting 40 features from all training images …")
        X, y, subclasses, paths = build_feature_dataset(
            args.data_dir, split="train", save_csv=feat_csv
        )

    counts = np.bincount(y, minlength=2)
    print(f"Normal : {counts[0]}")
    print(f"Cancerous : {counts[1]}")

    # counts = np.bincount(y, minlength=7)
    # print(f"Total samples : {len(y)}")
    # for i, cls in enumerate(SUBCLASS_NAMES):
    #     print(f"{cls:22s}: {counts[i]}")
    # print(f"Features/image: {X.shape[1]}")

    # ── Step 3: Classification (Table 3, Figure 5, Figure 6) ─────────────
    print("\n[Step 4] Running classification …")
    results = run_classification(X, y, save_models=True)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" SUMMARY (Best results per classifier type)")
    print("=" * 65)

    svm_res = [r for r in results if r["label"].startswith("SVM")]
    mlp_res = [r for r in results if r["label"].startswith("MLP")]

    best_svm = max(svm_res, key=lambda r: r["Acc"])
    best_mlp = max(mlp_res, key=lambda r: r["Acc"])

    print(f"\n  Best SVM → {best_svm['label']}")
    print(f"    Accuracy  : {best_svm['Acc']}%")
    print(f"    Sensitivity: {best_svm['Sen']}%")
    print(f"    Specificity: {best_svm['Spc']}%")
    print(f"    AUC       : {best_svm.get('AUC', 'N/A')}")

    print(f"\n  Best MLP → {best_mlp['label']}")
    print(f"    Accuracy  : {best_mlp['Acc']}%")
    print(f"    Sensitivity: {best_mlp['Sen']}%")
    print(f"    Specificity: {best_mlp['Spc']}%")
    print(f"    AUC       : {best_mlp.get('AUC', 'N/A')}")

    # best_model = max(results, key=lambda r: r["Acc"])
    # print(f"\nBest Overall Model : {best_model['label']}")
    # print(f"Accuracy           : {best_model['Acc']}%")

    print("Target: Highest possible 7-class accuracy")
    print("\n" + "=" * 65)
    print(f" All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
