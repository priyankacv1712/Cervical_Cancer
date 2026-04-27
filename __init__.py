# src/__init__.py
from .preprocessing      import load_image, apply_clahe, generate_figure2
from .segmentation       import afkm_segment, kmeans_segment, generate_figure3
from .feature_extraction import extract_features, build_feature_dataset, FEATURE_NAMES
from .classification     import run_classification, evaluate_classifier
from .predict            import predict_image


# new code
# src/__init__.py

# from .preprocessing import load_image, apply_clahe, generate_figure2
# from .segmentation import afkm_segment, kmeans_segment, generate_figure3
# from .feature_extraction import (
#     extract_features,
#     build_feature_dataset,
#     FEATURE_NAMES
# )
# from .classification import run_classification, evaluate_classifier
# from .predict import predict_image

# __all__ = [
#     "load_image",
#     "apply_clahe",
#     "generate_figure2",
#     "afkm_segment",
#     "kmeans_segment",
#     "generate_figure3",
#     "extract_features",
#     "build_feature_dataset",
#     "FEATURE_NAMES",
#     "run_classification",
#     "evaluate_classifier",
#     "predict_image",
# ]