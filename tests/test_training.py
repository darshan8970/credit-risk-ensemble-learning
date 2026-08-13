from src.train import train_and_tune_models


def test_training_function_is_available():
    assert callable(train_and_tune_models)