from typing import NoReturn

import pandas as pd

from src.data import read_data, split_train_val_data
from src.entities import SplittingParams


def test_read_data(dataset_path: str) -> NoReturn:
    df = read_data(dataset_path)
    expected_shape = (100, 14)
    assert isinstance(df, pd.DataFrame)
    assert expected_shape == df.shape


def test_split_train_val_data(dataset_path: str) -> NoReturn:
    df = read_data(dataset_path)
    val_size = 0.3
    params = SplittingParams(val_size=val_size, random_state=40)
    train_df, val_df = split_train_val_data(df, params)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert len(train_df) + len(val_df) == len(df)
