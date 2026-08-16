import numpy as np
import pytest

from src.predict import predict_credit_risk


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

def test_predict_credit_risk_with_multiple_samples():
    model = MockModel()

    features = [
        [1, 10],
        [2, 20],
        [3, 30],
    ]

    prediction = predict_credit_risk(model, features)

    assert prediction.tolist() == [1, 1, 1]