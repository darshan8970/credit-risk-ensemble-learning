import numpy as np
import pytest
import joblib

from src.predict import (
    predict_credit_risk,
    load_model,
    interpret_credit_risk,
    predict_from_saved_model,
    list_saved_models,
)


class MockModel:
    def predict(self, X):
        return np.ones(len(X), dtype=int)

class InvalidPredictionModel:
    def predict(self, X):
        return np.array([2])

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

def test_predict_credit_risk_rejects_empty_input():
    model = MockModel()

    with pytest.raises(ValueError, match="Prediction features cannot be empty"):
        predict_credit_risk(model, [])

def test_predict_from_saved_model(tmp_path, monkeypatch):
    model_path = tmp_path / "test_model.joblib"

    joblib.dump(MockModel(), model_path)

    monkeypatch.setattr("src.predict.MODELS_DIR", str(tmp_path))

    result = predict_from_saved_model(
        "test_model",
        [1, 10],
    )

    assert result == {
        "prediction": 1,
        "risk_label": "Default",
    }

def test_predict_from_saved_model_rejects_invalid_prediction(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "invalid_model.joblib"

    joblib.dump(InvalidPredictionModel(), model_path)

    monkeypatch.setattr("src.predict.MODELS_DIR", str(tmp_path))

    with pytest.raises(
        ValueError,
        match="Unsupported credit-risk prediction"
    ):
        predict_from_saved_model(
            "invalid_model",
            [1, 10],
        )

def test_list_saved_models(tmp_path, monkeypatch):
    (tmp_path / "random_forest_model.joblib").touch()
    (tmp_path / "bagging_model.joblib").touch()
    (tmp_path / "adaboost_model.joblib").touch()
    (tmp_path / "preprocessor.joblib").touch()
    (tmp_path / "notes.txt").touch()

    monkeypatch.setattr("src.predict.MODELS_DIR", str(tmp_path))

    assert list_saved_models() == [
        "adaboost_model",
        "bagging_model",
        "random_forest_model",
    ]