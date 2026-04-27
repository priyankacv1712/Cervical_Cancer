# segmentation.py
# Adaptive Fuzzy K-Means (AFKM) and standard K-Means segmentation
# Implements paper Section 2.1, equations (1) and (2), Figure 3

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (KMEANS_N_CLUSTERS, KMEANS_MAX_ITER,
                    AFKM_FUZZY_M, AFKM_MAX_ITER, AFKM_TOL,
                    AFKM_N_CLUSTERS, FIG3_PATH, OUTPUT_DIR, IMG_SIZE)
from preprocessing import load_image, apply_clahe


# ─────────────────────────────────────────────────────────────────────────────
# Standard K-Means segmentation
# ─────────────────────────────────────────────────────────────────────────────
def kmeans_segment(img_bgr: np.ndarray, n_clusters: int = KMEANS_N_CLUSTERS):
    """
    Standard OpenCV K-Means clustering for image segmentation.
    Returns the segmented label map and the colour-coded image.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pixels   = img_gray.reshape(-1, 1).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                KMEANS_MAX_ITER, 1e-4)
    _, labels, centres = cv2.kmeans(pixels, n_clusters, None, criteria,
                                    10, cv2.KMEANS_RANDOM_CENTERS)

    # Sort clusters by intensity (background=0, cytoplasm=1, nucleus=2)
    order   = np.argsort(centres.flatten())
    remap   = np.zeros(n_clusters, dtype=np.uint8)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx

    labels_remapped = remap[labels.flatten()].reshape(img_gray.shape)

    # Colour-coded output (red-ish palette like paper Figure 3b)
    colour_map = np.array([[0, 0, 0],
                            [180, 50, 50],
                            [220, 200, 50]], dtype=np.uint8)
    coloured = colour_map[labels_remapped]

    # Binary mask of foreground (nucleus + cytoplasm)
    binary = (labels_remapped > 0).astype(np.uint8) * 255
    return labels_remapped, coloured, binary


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Fuzzy K-Means (AFKM) segmentation
# Paper: Section 2.1, equations (1)–(2)
# Reference: Isa et al., IEEE Trans. Consumer Electronics, 2009
# ─────────────────────────────────────────────────────────────────────────────
def _afkm_init_centres(pixels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Initialise cluster centres using evenly-spaced intensity percentiles."""
    percentiles = np.linspace(0, 100, n_clusters + 2)[1:-1]
    return np.array([np.percentile(pixels, p) for p in percentiles],
                    dtype=np.float64).reshape(n_clusters, -1)


def afkm_segment(img_bgr: np.ndarray,
                 n_clusters: int  = AFKM_N_CLUSTERS,
                 m: float         = AFKM_FUZZY_M,
                 max_iter: int    = AFKM_MAX_ITER,
                 tol: float       = AFKM_TOL):
    """
    Adaptive Fuzzy K-Means segmentation as described in the paper.

    Equations implemented:
      (1)  C_j  = (1/m_Cj) * Σ l(x,y)   for all pixels in cluster j
      (2)  m_jp = 1 / Σ_k [ (d_jp/d_kp)^(2/(m-1)) ]

    The 'adaptive' part: degree of belongingness B_j = C_j / m_jp
    and membership reassignment with constraint l(x,y) < c_larg
    to avoid mis-clustering of the brightest cluster.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w     = img_gray.shape
    pixels   = img_gray.reshape(-1, 1).astype(np.float64)
    N        = pixels.shape[0]

    # Initialise
    centres = _afkm_init_centres(pixels, n_clusters)   # (K, 1)
    exp     = 2.0 / (m - 1)

    for iteration in range(max_iter):
        # ── Step 1: compute distances  d_jp  ─────────────────────────────
        # shape: (N, K)
        dists = np.abs(pixels - centres.T)              # broadcast
        dists = np.maximum(dists, 1e-10)                # avoid /0

        # ── Step 2: membership matrix  m_jp  (eq. 2) ─────────────────────
        ratio = dists[:, :, np.newaxis] / dists[:, np.newaxis, :]  # (N,K,K)
        membership = 1.0 / np.sum(ratio ** exp, axis=2)            # (N,K)

        # ── Step 3: update cluster centres  C_j  (eq. 1) ─────────────────
        new_centres = np.zeros_like(centres)
        for j in range(n_clusters):
            mj  = membership[:, j]                      # (N,)
            # Degree of belongingness B_j = C_j / m_jp → adaptive weight
            Bj  = centres[j, 0] / (mj + 1e-10)
            # Constraint: only include pixels < c_larg (largest centre)
            c_larg = centres.max()
            mask = (pixels[:, 0] < c_larg).astype(np.float64)
            weights = mj * mask
            denom   = weights.sum()
            if denom > 1e-10:
                new_centres[j, 0] = np.dot(weights, pixels[:, 0]) / denom
            else:
                new_centres[j, 0] = centres[j, 0]

        # ── Convergence check ─────────────────────────────────────────────
        delta = np.max(np.abs(new_centres - centres))
        centres = new_centres
        if delta < tol:
            break

    # Hard assignment
    labels = np.argmax(membership, axis=1).reshape(h, w)

    # Sort by intensity
    order  = np.argsort(centres.flatten())
    remap  = np.zeros(n_clusters, dtype=np.uint8)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    labels = remap[labels]

    # Binary foreground mask (nucleus + cytoplasm = labels > 0)
    binary = (labels > 0).astype(np.uint8) * 255

    # Black-and-white output (as in Figure 3d of the paper)
    visual        = np.zeros((h, w), dtype=np.uint8)
    visual[labels == 0] = 0          # background  → black
    visual[labels == 1] = 128        # cytoplasm   → grey
    visual[labels == 2] = 255        # nucleus     → white
    visual_bgr = cv2.cvtColor(visual, cv2.COLOR_GRAY2BGR)

    return labels, visual_bgr, binary


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-colour helper (Figure 3c)
# ─────────────────────────────────────────────────────────────────────────────
def pseudo_colour(labels: np.ndarray) -> np.ndarray:
    """Map integer label map to a blue-red pseudo-colour image (paper Fig 3c)."""
    n = labels.max() + 1
    palette = np.array([
        [0,   0,   0],    # background → black
        [0,   0, 200],    # cytoplasm  → blue
        [200, 50,  50],   # nucleus    → red
    ], dtype=np.uint8)
    if n > len(palette):
        extra = np.random.randint(50, 200, (n - len(palette), 3), dtype=np.uint8)
        palette = np.vstack([palette, extra])
    return palette[labels]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure3(image_paths: list, save_path: str = FIG3_PATH) -> None:
    """
    Reproduce Figure 3 of the paper (4 rows × 4 columns):
      (a) original input
      (b) K-Means output
      (c) Pseudo-colour of K-Means
      (d) AFKM output
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_rows = min(4, len(image_paths))
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    col_titles = [
        "(a) Input image",
        "(b) K-Means output",
        "(c) Pseudo-colour output",
        "(d) AFKM output",
    ]
    row_labels = [f"{i+1}" for i in range(n_rows)]

    for i, path in enumerate(image_paths[:n_rows]):
        img_bgr  = load_image(path)
        img_enh  = apply_clahe(img_bgr)

        km_labels, km_coloured, _   = kmeans_segment(img_enh)
        pc_img                       = pseudo_colour(km_labels)
        afkm_labels, afkm_vis, _    = afkm_segment(img_enh)

        images = [
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(km_coloured, cv2.COLOR_BGR2RGB),
            pc_img,
            cv2.cvtColor(afkm_vis, cv2.COLOR_BGR2RGB),
        ]

        for j, (ax, img) in enumerate(zip(axes[i], images)):
            ax.imshow(img)
            ax.axis("off")
            # column headers on first row, row labels on first column
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10, pad=6)
            if j == 0:
                ax.set_ylabel(f"a{i+1} / b{i+1} / c{i+1} / d{i+1}",
                               fontsize=8, rotation=0, labelpad=50, va="center")

    plt.suptitle(
        "Figure 3. ROI extractions by k-means clustering and AFKM algorithm;\n"
        "(a) input image, (b) k-means clustering output, "
        "(c) pseudo colour output image, and (d) AFKM output",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure 3] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from preprocessing import collect_image_paths

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="herlev dataset original")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = collect_image_paths(args.data_dir, "train")
    print(f"Found {len(paths)} training images.")

    # One image per sub-class (4 total for the figure)
    sample_paths = []
    seen_dirs = set()
    for p in paths:
        d = os.path.basename(os.path.dirname(p))
        if d not in seen_dirs:
            sample_paths.append(p)
            seen_dirs.add(d)
        if len(sample_paths) == 4:
            break

    generate_figure3(sample_paths)
