import os
import joblib
import numpy as np

from src.config import MODELS_DIR


def load_model(model_name: str):
    """
    Load a trained model from the models directory.

    Parameters:
        model_name: Name of the saved model file without .joblib

    Returns:
        Loaded model instance.
    """
    filename = f"{model_name}.joblib"
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    return joblib.load(model_path)


def predict_credit_risk(model, features):
    """
    Generate a credit-risk prediction using a trained model.

    Parameters:
        model: Trained classification model.
        features: Input feature array.

    Returns:
        Model prediction.
    """
    features = np.asarray(features)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    return model.predict(features)