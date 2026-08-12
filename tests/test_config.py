from src.config import PARAM_GRIDS


def test_model_parameter_grids_exist():
    expected_models = {"Random Forest", "Bagging", "AdaBoost"}

    assert set(PARAM_GRIDS.keys()) == expected_models


def test_random_forest_grid():
    grid = PARAM_GRIDS["Random Forest"]

    assert "n_estimators" in grid
    assert "max_depth" in grid
    assert "min_samples_split" in grid
    assert "class_weight" in grid


def test_bagging_grid():
    grid = PARAM_GRIDS["Bagging"]

    assert "n_estimators" in grid
    assert "max_samples" in grid
    assert "max_features" in grid


def test_adaboost_grid():
    grid = PARAM_GRIDS["AdaBoost"]

    assert "n_estimators" in grid
    assert "learning_rate" in grid