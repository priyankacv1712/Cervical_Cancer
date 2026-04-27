# preprocessing.py
# CLAHE pre-processing exactly as described in the paper (Section 3, Figure 2)

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, IMG_SIZE,
                    VALID_EXTENSIONS, FIG2_PATH, OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    """Load an image as BGR (OpenCV default)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, IMG_SIZE)
    return img


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    # """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to each
    # channel of a BGR image, as described in the paper.

    # The paper applies CLAHE to increase brightness and facilitate finding
    # visual details, improving segmentation boundary definition.
    # """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                             tileGridSize=CLAHE_TILE_GRID)
    channels = cv2.split(img_bgr)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)


    # def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    # # """
    # # Enhanced preprocessing for highest accuracy:
    # # 1. Denoise
    # # 2. LAB CLAHE on luminance
    # # 3. Sharpen lightly
    # # """

    # # Denoise
    # img_bgr = cv2.fastNlMeansDenoisingColored(
    #     img_bgr, None, 5, 5, 7, 21
    # )

    # # Convert to LAB
    # lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)

    # # CLAHE only on brightness channel
    # clahe = cv2.createCLAHE(
    #     clipLimit=CLAHE_CLIP_LIMIT,
    #     tileGridSize=CLAHE_TILE_GRID
    # )

    # l = clahe.apply(l)

    # lab = cv2.merge((l, a, b))
    # enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # # Mild sharpening
    # kernel = np.array([
    #     [0,-1,0],
    #     [-1,5,-1],
    #     [0,-1,0]
    # ])

    # enhanced = cv2.filter2D(enhanced, -1, kernel)

    # return enhanced


# ─────────────────────────────────────────────────────────────────────────────
def generate_figure2(image_paths: list, save_path: str = FIG2_PATH) -> None:
    """
    Reproduce Figure 2 of the paper:
    Shows 4 pairs of (original, CLAHE-enhanced) images side-by-side.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = min(4, len(image_paths))
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]

    for i, path in enumerate(image_paths[:n]):
        img_bgr  = load_image(path)
        img_enh  = apply_clahe(img_bgr)
        img_rgb  = cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2RGB)
        img_enh_rgb = cv2.cvtColor(img_enh, cv2.COLOR_BGR2RGB)

        axes[i][0].imshow(img_rgb)
        axes[i][0].set_title(f"(a) Input image", fontsize=12)
        axes[i][0].axis("off")

        axes[i][1].imshow(img_enh_rgb)
        axes[i][1].set_title(f"(b) Output image (CLAHE)", fontsize=12)
        axes[i][1].axis("off")

    plt.suptitle("Figure 2. Pre-processing using CLAHE", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure 2] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def collect_image_paths(data_dir: str, split: str = "train") -> list:
    """Walk a split folder and return all valid image paths."""
    split_dir = os.path.join(data_dir, split)
    paths = []
    if not os.path.isdir(split_dir):
        # flat directory fallback
        split_dir = data_dir
    for root, _, files in os.walk(split_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="herlev dataset original")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = collect_image_paths(args.data_dir, "train")
    print(f"Found {len(paths)} training images.")

    # Pick one image from each of 4 different sub-classes for a nice figure
    sample_paths = []
    seen_dirs = set()
    for p in paths:
        d = os.path.basename(os.path.dirname(p))
        if d not in seen_dirs:
            sample_paths.append(p)
            seen_dirs.add(d)
        if len(sample_paths) == 4:
            break

    generate_figure2(sample_paths)
