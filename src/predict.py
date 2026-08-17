import os
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from src.config import MODELS_DIR


def load_model(model_name: str) -> Any:
    """
    Load a trained model from the models directory.

    Parameters:
        model_name: Name of the saved model file without .joblib.

    Returns:
        Loaded model instance.

    Raises:
        FileNotFoundError: If the requested model does not exist.
    """
    filename = f"{model_name}.joblib"
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    return joblib.load(model_path)


def predict_credit_risk(
    model: Any,
    features: NDArray
) -> NDArray:
    """
    Generate a credit-risk prediction using a trained model.

    Parameters:
        model: Trained classification model.
        features: Input feature array.

    Returns:
        Model predictions.

    Raises:
        ValueError: If the input features are empty.
    """
    features = np.asarray(features)

    if features.size == 0:
        raise ValueError("Prediction features cannot be empty.")

    if features.ndim == 1:
        features = features.reshape(1, -1)

    return model.predict(features)


def interpret_credit_risk(prediction: NDArray) -> str:
    """
    Convert a numeric credit-risk prediction into a readable label.

    Parameters:
        prediction: Numeric prediction returned by a classification model.

    Returns:
        Human-readable credit-risk label.

    Raises:
        ValueError: If the prediction is not 0 or 1.
    """
    prediction = int(np.asarray(prediction).ravel()[0])

    labels = {
        0: "Non-Default",
        1: "Default",
    }

    if prediction not in labels:
        raise ValueError(
            f"Unsupported credit-risk prediction: {prediction}"
        )

    return labels[prediction]


def predict_from_saved_model(
    model_name: str,
    features: NDArray
) -> dict[str, Any]:
    """
    Load a saved model and generate a human-readable
    credit-risk prediction.

    Parameters:
        model_name: Name of the saved model file without .joblib.
        features: Input feature array.

    Returns:
        Dictionary containing the numeric prediction
        and readable risk label.
    """
    model = load_model(model_name)
    prediction = predict_credit_risk(model, features)
    risk_label = interpret_credit_risk(prediction)

    return {
        "prediction": int(np.asarray(prediction).ravel()[0]),
        "risk_label": risk_label,
    }