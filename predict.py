# predict.py
# Classify a single Pap smear image using saved models

import os
import sys
import argparse
import joblib
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from config    import MODELS_DIR, OUTPUT_DIR, IMG_SIZE
from preprocessing     import load_image, apply_clahe
from feature_extraction import extract_features

from config import MODELS_DIR, SUBCLASS_NAMES

def predict_image(image_path: str,
                  model_path: str = None,
                  show_steps: bool = True) -> dict:
    """
    Predict class of a single cervical cell image.

    Returns dict with:
      - prediction  : "Normal" or "Cancerous"
      - probability : confidence score (0–1)
      - features    : extracted 40-feature vector
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "svm_best.pkl")
        # model_path = os.path.join(MODELS_DIR, "best_model.pkl")


    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run main.py first to train and save models."
        )

    model = joblib.load(model_path)

    img   = load_image(image_path)
    feats = extract_features(img).reshape(1, -1)

    pred      = model.predict(feats)[0]
    label_str = "Normal" if pred == 0 else "Cancerous"
    # pred = int(model.predict(feats)[0])
    # label_str = SUBCLASS_NAMES[pred]

    prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats)[0]
        prob  = float(proba[pred])

    if show_steps:
        print(f"Image     : {image_path}")
        print(f"Prediction: {label_str}")
        # print(f"Predicted Class : {label_str}")
        if prob is not None:
            print(f"Confidence: {prob:.3f}")

    return {
        "prediction":  label_str,
        "probability": prob,
        "features":    feats[0].tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict cervical cell class from a single image"
    )
    parser.add_argument("image", type=str,
                        help="Path to input Pap smear image")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model .pkl (default: svm_best.pkl)")
    args = parser.parse_args()

    result = predict_image(args.image, args.model)
    print(f"\nResult: {result['prediction']}")


