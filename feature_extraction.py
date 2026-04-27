# feature_extraction.py
# Extracts the exact 40 features described in paper Section 2.2
# 20 nucleus features + 20 cytoplasm features

import cv2
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import (CLASS_MAP, SUBCLASS_NAMES, VALID_EXTENSIONS,
                    GLCM_DISTANCES, GLCM_ANGLES, FEATURES_CSV, OUTPUT_DIR,
                    IMG_SIZE)
from preprocessing import load_image, apply_clahe, collect_image_paths
from segmentation  import afkm_segment

# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.models import Model

# cnn_base = EfficientNetB0(
#     weights="imagenet",
#     include_top=False,
#     pooling="avg",
#     input_shape=(224,224,3)
# )

# ─────────────────────────────────────────────────────────────────────────────
# GLCM (Gray-Level Co-occurrence Matrix) helpers
# ─────────────────────────────────────────────────────────────────────────────
def _glcm(gray_region: np.ndarray, levels: int = 256) -> np.ndarray:
    """
    Compute a normalised symmetric GLCM averaged over 4 angles.
    Returns (levels×levels) probability matrix.
    """
    h, w   = gray_region.shape
    glcm   = np.zeros((levels, levels), dtype=np.float64)
    dxdy   = [(1, 0), (1, 1), (0, 1), (-1, 1)]   # 0°,45°,90°,135°

    for dx, dy in dxdy:
        for y in range(h):
            for x in range(w):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    i = int(gray_region[y, x])
                    j = int(gray_region[ny, nx])
                    glcm[i, j] += 1
                    glcm[j, i] += 1   # symmetric

    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def _glcm_features(gray_region: np.ndarray) -> dict:
    """
    Extract GLCM-based texture features: contrast, correlation,
    variance, energy, entropy, probability.
    Paper uses these for both nucleus and cytoplasm regions.
    """
    # Downsample levels to 32 for speed while retaining accuracy
    levels  = 32
    scaled  = (gray_region / 8).astype(np.uint8).clip(0, levels - 1)
    glcm    = _glcm(scaled, levels=levels)

    i_vals  = np.arange(levels)
    j_vals  = np.arange(levels)
    I, J    = np.meshgrid(i_vals, j_vals, indexing="ij")

    mu_i    = np.sum(I * glcm)
    mu_j    = np.sum(J * glcm)
    sigma_i = np.sqrt(np.sum((I - mu_i) ** 2 * glcm) + 1e-10)
    sigma_j = np.sqrt(np.sum((J - mu_j) ** 2 * glcm) + 1e-10)

    contrast    = float(np.sum((I - J) ** 2 * glcm))
    correlation = float(np.sum((I - mu_i) * (J - mu_j) * glcm) /
                        (sigma_i * sigma_j + 1e-10))
    variance    = float(np.sum((I - mu_i) ** 2 * glcm))
    energy      = float(np.sum(glcm ** 2))
    entropy     = float(-np.sum(glcm * np.log2(glcm + 1e-10)))
    probability = float(glcm.max())

    return {
        "contrast":    contrast,
        "correlation": correlation,
        "variance":    variance,
        "energy":      energy,
        "entropy":     entropy,
        "probability": probability,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Region-property helpers
# ─────────────────────────────────────────────────────────────────────────────
def _region_props(binary_mask: np.ndarray) -> dict:
    """
    Compute morphological/shape features from a binary mask.
    Matches the nucleus/cytoplasm features listed in paper Section 2.2.
    """
    props = {
        "area": 0.0, "major_minor_ratio": 0.0,
        "diameter": 0.0, "short_long_ratio": 0.0,
        "extent": 0.0, "eccentricity": 0.0,
        "perimeter": 0.0, "elongation": 0.0, "orientation": 0.0,
    }

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return props

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 4:
        return props

    perimeter = cv2.arcLength(cnt, True)
    x, y, bw, bh = cv2.boundingRect(cnt)
    extent = area / (bw * bh + 1e-10)

    if len(cnt) >= 5:
        ellipse          = cv2.fitEllipse(cnt)
        (cx, cy), (MA, ma), angle = ellipse
        MA               = max(MA, 1e-3)
        ma               = max(ma, 1e-3)
        major_axis       = max(MA, ma)
        minor_axis       = min(MA, ma)
        major_minor_ratio= major_axis / (minor_axis + 1e-10)
        short_long_ratio = minor_axis / (major_axis + 1e-10)
        elongation       = (major_axis - minor_axis) / (major_axis + 1e-10)
        diameter         = (MA + ma) / 2.0
        # Eccentricity of best-fit ellipse
        c_dist           = np.sqrt(max(major_axis**2 - minor_axis**2, 0))
        eccentricity     = c_dist / (major_axis + 1e-10)
        orientation      = float(angle)
    else:
        major_minor_ratio = 1.0
        short_long_ratio  = 1.0
        elongation        = 0.0
        diameter          = np.sqrt(area / np.pi) * 2
        eccentricity      = 0.0
        orientation       = 0.0

    props.update({
        "area":              float(area),
        "major_minor_ratio": float(major_minor_ratio),
        "diameter":          float(diameter),
        "short_long_ratio":  float(short_long_ratio),
        "extent":            float(extent),
        "eccentricity":      float(eccentricity),
        "perimeter":         float(perimeter),
        "elongation":        float(elongation),
        "orientation":       float(orientation),
    })
    return props


def _colour_intensity(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """Mean intensity in R, G, B channels within masked region."""
    if mask.sum() == 0:
        return {"mean_R": 0.0, "mean_G": 0.0, "mean_B": 0.0}
    b, g, r = cv2.split(img_bgr)
    return {
        "mean_R": float(r[mask > 0].mean()),
        "mean_G": float(g[mask > 0].mean()),
        "mean_B": float(b[mask > 0].mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main feature extraction function  →  40 features
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full 40-feature extraction pipeline for one image.

    Nucleus features (20):
      area, major_minor_ratio, diameter, short_long_ratio, extent,
      eccentricity, perimeter, elongation, orientation,
      mean_R, mean_G, mean_B,
      contrast, correlation, variance, energy, entropy, probability,
      nucleus_position_x, nucleus_position_y

    Cytoplasm features (20):
      area, major_minor_ratio, diameter, short_long_ratio, extent,
      eccentricity, perimeter, elongation, orientation,
      nucleus_cytoplasm_ratio, mean_R, mean_G, mean_B,
      contrast, correlation, variance, energy, entropy, probability,
      (orientation reused as cytoplasm_orientation)

    Returns: np.ndarray of shape (40,)
    """
    img_enh = apply_clahe(img_bgr)
    labels, _, binary = afkm_segment(img_enh)

    # ── Nucleus mask (label == 2, brightest cluster) ──────────────────────
    nucleus_mask   = (labels == 2).astype(np.uint8) * 255
    # ── Cytoplasm mask (label == 1) ───────────────────────────────────────
    cytoplasm_mask = (labels == 1).astype(np.uint8) * 255

    img_gray = cv2.cvtColor(img_enh, cv2.COLOR_BGR2GRAY)

    # ── Nucleus features ──────────────────────────────────────────────────
    nuc_props  = _region_props(nucleus_mask)
    nuc_colour = _colour_intensity(img_bgr, nucleus_mask)

    # Position of nucleus centroid (normalised)
    cnts, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        M  = cv2.moments(max(cnts, key=cv2.contourArea))
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"] / img_gray.shape[1]
            cy = M["m01"] / M["m00"] / img_gray.shape[0]
        else:
            cx = cy = 0.5
    else:
        cx = cy = 0.5

    nuc_region = img_gray.copy()
    nuc_region[nucleus_mask == 0] = 0
    nuc_texture = _glcm_features(nuc_region)

    nucleus_feats = [
        nuc_props["area"],
        nuc_props["major_minor_ratio"],
        nuc_props["diameter"],
        nuc_props["short_long_ratio"],
        nuc_props["extent"],
        nuc_props["eccentricity"],
        nuc_props["perimeter"],
        nuc_props["elongation"],
        nuc_props["orientation"],
        nuc_colour["mean_R"],
        nuc_colour["mean_G"],
        nuc_colour["mean_B"],
        nuc_texture["contrast"],
        nuc_texture["correlation"],
        nuc_texture["variance"],
        nuc_texture["energy"],
        nuc_texture["entropy"],
        nuc_texture["probability"],
        cx,          # position_x
        cy,          # position_y
    ]

    # ── Cytoplasm features ────────────────────────────────────────────────
    cyt_props  = _region_props(cytoplasm_mask)
    cyt_colour = _colour_intensity(img_bgr, cytoplasm_mask)

    nuc_area = nuc_props["area"]
    cyt_area = cyt_props["area"]
    nc_ratio = nuc_area / (cyt_area + 1e-10)   # nucleus-to-cytoplasm ratio

    cyt_region = img_gray.copy()
    cyt_region[cytoplasm_mask == 0] = 0
    cyt_texture = _glcm_features(cyt_region)

    cytoplasm_feats = [
        cyt_props["area"],
        cyt_props["major_minor_ratio"],
        cyt_props["diameter"],
        cyt_props["short_long_ratio"],
        cyt_props["extent"],
        cyt_props["eccentricity"],
        cyt_props["perimeter"],
        cyt_props["elongation"],
        cyt_props["orientation"],
        nc_ratio,             # nucleus_cytoplasm_ratio (paper specific feature)
        cyt_colour["mean_R"],
        cyt_colour["mean_G"],
        cyt_colour["mean_B"],
        cyt_texture["contrast"],
        cyt_texture["correlation"],
        cyt_texture["variance"],
        cyt_texture["energy"],
        cyt_texture["entropy"],
        cyt_texture["probability"],
        cyt_props["elongation"],   # 20th: cytoplasm elongation (C9 in paper)
    ]

    all_feats = nucleus_feats + cytoplasm_feats   # 40 features total
    return np.array(all_feats, dtype=np.float64)
    # # handcrafted features
    # handcrafted = np.array(nucleus_feats + cytoplasm_feats, dtype=np.float64)

    # # deep features
    # img224 = cv2.resize(img_bgr, (224,224))
    # x = np.expand_dims(img224.astype(np.float32), axis=0)
    # x = preprocess_input(x)

    # deep_feats = cnn_base.predict(x, verbose=0)[0]

    # # combine both
    # all_feats = np.concatenate([handcrafted, deep_feats])

    # return all_feats


# ─────────────────────────────────────────────────────────────────────────────
# Feature column names (for DataFrame)
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE_NAMES = [f"feat_{i}" for i in range(1320)]
FEATURE_NAMES = (
    [f"nuc_{n}" for n in [
        "area", "major_minor_ratio", "diameter", "short_long_ratio",
        "extent", "eccentricity", "perimeter", "elongation", "orientation",
        "mean_R", "mean_G", "mean_B",
        "contrast", "correlation", "variance", "energy", "entropy",
        "probability", "position_x", "position_y"
    ]] +
    [f"cyt_{n}" for n in [
        "area", "major_minor_ratio", "diameter", "short_long_ratio",
        "extent", "eccentricity", "perimeter", "elongation", "orientation",
        "nc_ratio",
        "mean_R", "mean_G", "mean_B",
        "contrast", "correlation", "variance", "energy", "entropy",
        "probability", "elongation2"
    ]]
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level extraction
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_dataset(data_dir: str,
                           split: str = "train",
                           save_csv: str = None) -> tuple:
    """
    Process all images in a split, extract 40 features each.
    Returns (X: np.ndarray [N,40], y: np.ndarray [N], subclass_y: [N], paths: [N])
    Also saves a CSV if save_csv is given.
    """
    from config import CLASS_MAP, VALID_EXTENSIONS

    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        split_dir = data_dir

    all_feats, labels, subclass_labels, file_paths = [], [], [], []

    sub_dirs = sorted([d for d in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, d))])

    if not sub_dirs:
        # flat structure
        sub_dirs = ["."]

    for sub in sub_dirs:
        sub_path  = os.path.join(split_dir, sub)
        class_key = sub.lower().replace(" ", "_")
        label     = CLASS_MAP.get(class_key, -1)

        img_files = [f for f in sorted(os.listdir(sub_path))
                     if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]

        for fname in tqdm(img_files, desc=f"  {sub}", leave=False):
            fpath = os.path.join(sub_path, fname)
            try:
                img  = load_image(fpath)
                feat = extract_features(img)
                all_feats.append(feat)
                labels.append(label)
                subclass_labels.append(class_key)
                file_paths.append(fpath)
            except Exception as e:
                print(f"  [WARN] Skipping {fpath}: {e}")

    X  = np.array(all_feats, dtype=np.float64)
    y  = np.array(labels,    dtype=np.int32)

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df.insert(0, "subclass",  subclass_labels)
        df.insert(0, "label",     y)
        df.insert(0, "filepath",  file_paths)
        df.to_csv(save_csv, index=False)
        print(f"[Features] Saved CSV → {save_csv}  (shape: {X.shape})")

    return X, y, subclass_labels, file_paths


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="herlev dataset original")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Extracting features from training set …")
    X, y, subs, paths = build_feature_dataset(
        args.data_dir, split="train", save_csv=FEATURES_CSV
    )
    print(f"Done. Feature matrix shape: {X.shape}, Labels: {np.bincount(y)}")
