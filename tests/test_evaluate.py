import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.evaluate import evaluate_models


class MockModel:
    feature_importances_ = np.array([0.6, 0.4])

    def predict(self, X):
        return np.array([0, 1, 0, 1])

    def predict_proba(self, X):
        return np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
        ])


def test_evaluation_returns_metrics_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluate.REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr("src.evaluate.ASSETS_DIR", str(tmp_path))

    X_test = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
    ])

    y_test = np.array([0, 1, 0, 1])

    trained_models = {
        "Random Forest": MockModel()
    }

    feature_names = ["feature_1", "feature_2"]

    result = evaluate_models(
        trained_models,
        X_test,
        y_test,
        feature_names,
    )

    assert not result.empty
    assert "Model" in result.columns
    assert "Accuracy" in result.columns
    assert "Precision" in result.columns
    assert "Recall" in result.columns
    assert "F1-Score" in result.columns
    assert "ROC-AUC" in result.columns
    assert "PR-AUC" in result.columns

    assert result.iloc[0]["Model"] == "Random Forest"

def test_evaluation_identifies_best_model_by_roc_auc(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("src.evaluate.REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr("src.evaluate.ASSETS_DIR", str(tmp_path))

    class WeakMockModel(MockModel):
        def predict_proba(self, X):
            return np.array([
                [0.40, 0.60],
                [0.60, 0.40],
                [0.45, 0.55],
                [0.55, 0.45],
            ])

    X_test = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
    ])

    y_test = np.array([0, 1, 0, 1])

    trained_models = {
        "Random Forest": WeakMockModel(),
        "AdaBoost": MockModel(),
    }

    feature_names = ["feature_1", "feature_2"]

    result = evaluate_models(
        trained_models,
        X_test,
        y_test,
        feature_names,
    )

    best_model = result.loc[
        result["ROC-AUC"].idxmax(),
        "Model"
    ]

    assert best_model == "AdaBoost"