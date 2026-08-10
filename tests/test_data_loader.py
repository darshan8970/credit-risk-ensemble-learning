import pandas as pd

from src.config import COLUMN_NAMES, TARGET_COLUMN
from src.data_loader import load_raw_dataset


def test_dataset_loading():
    df = load_raw_dataset()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_expected_columns_exist():
    df = load_raw_dataset()

    assert list(df.columns) == COLUMN_NAMES
    assert TARGET_COLUMN in df.columns


def test_target_is_binary():
    df = load_raw_dataset()

    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})