import joblib
import numpy as np
import main


class MockModel:
    def predict(self, X):
        return np.array([0, 1])


def test_main_module_imports():
    assert callable(main.main)


def test_saved_model_can_be_loaded(tmp_path):
    model = MockModel()

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    loaded_model = joblib.load(model_path)

    predictions = loaded_model.predict(
        np.array([
            [1, 10],
            [2, 20],
        ])
    )

    assert len(predictions) == 2
    assert predictions.tolist() == [0, 1]