import os
from typing import NoReturn, List

import pytest
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

from src.data import read_data
from src.features import (
    extract_target,
    build_transformer,
    make_features,
    serialize_transformer,
    deserialize_transformer,
)
from src.entities import FeatureParams
from src.features import MyStandardScaler
from tests.fake_data import generate_fake_int_matrix


@pytest.fixture(scope="package")
def feature_params(
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    fp = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    return fp


def test_extract_target(dataset_path: str, feature_params: FeatureParams) -> NoReturn:
    df = read_data(dataset_path)
    target_df = extract_target(df, feature_params)
    assert isinstance(target_df, pd.Series)
    assert len(target_df) == len(df)
    assert df[feature_params.target_col].equals(target_df)


def test_custom_transformer() -> NoReturn:
    arr = generate_fake_int_matrix(num_rows=5, num_cols=10)
    expected_arr = (arr - arr.mean(axis=0)) / arr.std(axis=0)
    scaler = MyStandardScaler()
    scaler.fit(arr)
    transformed_arr = scaler.transform(arr)
    assert arr.shape == transformed_arr.shape
    assert np.allclose(expected_arr, transformed_arr)


def test_column_transformer(
    fake_pd_dataframe: pd.DataFrame, feature_params: FeatureParams
) -> NoReturn:
    transformer = build_transformer(feature_params)
    transformer.fit(fake_pd_dataframe)
    check_is_fitted(transformer)
    transformed_fake_pd_dataframe = make_features(transformer, fake_pd_dataframe)
    expected_rows = fake_pd_dataframe.shape[0]
    expected_cols = 30
    assert not pd.isnull(transformed_fake_pd_dataframe).any().any()
    assert isinstance(transformed_fake_pd_dataframe, pd.DataFrame)
    assert (
        expected_rows,
        expected_cols,
    ) == transformed_fake_pd_dataframe.shape


def test_serialize_and_deserialize_transformer(
    fake_pd_dataframe: pd.DataFrame,
    feature_params: FeatureParams,
    transformer_path: str,
) -> NoReturn:
    transformer = build_transformer(feature_params)
    transformer.fit(fake_pd_dataframe)
    serialize_transformer(transformer, transformer_path)
    transformed_fake_pd_dataframe = make_features(transformer, fake_pd_dataframe)
    assert os.path.exists(transformer_path)
    deserialized_transformer = deserialize_transformer(transformer_path)
    deserialized_transformed_fake_pd_dataframe = make_features(
        deserialized_transformer, fake_pd_dataframe
    )
    assert not pd.isnull(deserialized_transformed_fake_pd_dataframe).any().any()
    assert isinstance(deserialized_transformed_fake_pd_dataframe, pd.DataFrame)
    assert deserialized_transformed_fake_pd_dataframe.equals(
        transformed_fake_pd_dataframe
    )
