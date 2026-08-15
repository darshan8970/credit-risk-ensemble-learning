import numpy as np

from src.predict import predict_credit_risk


class MockModel:
    def predict(self, X):
        return np.array([1])


def test_predict_credit_risk():
    model = MockModel()

    features = [1, 10]

    prediction = predict_credit_risk(model, features)

    assert prediction.tolist() == [1]