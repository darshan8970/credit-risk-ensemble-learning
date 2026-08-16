import numpy as np
import pytest

from src.predict import (
    predict_credit_risk,
    load_model,
    interpret_credit_risk,
)


class MockModel:
    def predict(self, X):
        return np.ones(len(X), dtype=int)


def test_predict_credit_risk():
    model = MockModel()

    features = [1, 10]

    prediction = predict_credit_risk(model, features)

    assert prediction.tolist() == [1]

def test_load_model_raises_error_for_missing_model(tmp_path, monkeypatch):
    monkeypatch.setattr("src.predict.MODELS_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Saved model not found"):
        from src.predict import load_model

        load_model("nonexistent_model")

def test_interpret_credit_risk():
    assert interpret_credit_risk([0]) == "Non-Default"
    assert interpret_credit_risk([1]) == "Default"


def test_interpret_credit_risk_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unsupported credit-risk prediction"):
        interpret_credit_risk([2])