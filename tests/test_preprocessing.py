import pandas as pd

from src.config import TARGET_COLUMN
from src.data_loader import load_raw_dataset
from src.preprocessing import prepare_data


def test_preprocessing_split_and_transform():
    df = load_raw_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        feature_names,
    ) = prepare_data(df)

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] == len(feature_names)

    assert TARGET_COLUMN not in feature_names