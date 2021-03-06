import os
from typing import NoReturn, List, Tuple

import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from src.entities import FeatureParams, RFParams
from src.data import read_data
from src.features import build_transformer, make_features, extract_target
from src.models import train_model, serialize_model, deserialize_model


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


@pytest.fixture(scope="package")
def training_params() -> RFParams:
    tp = RFParams(
        model_type="RandomForestClassifier",
        n_estimators=90,
        criterion="gini",
        random_state=5,
    )
    return tp


@pytest.fixture(scope="package")
def preprocess_data(
    dataset_path: str, feature_params: FeatureParams
) -> Tuple[pd.Series, pd.DataFrame]:
    df = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(df)
    transformed_features = make_features(transformer, df)
    target = extract_target(df, feature_params)
    return target, transformed_features


def test_train_model(
    training_params: RFParams,
    preprocess_data: Tuple[pd.Series, pd.DataFrame],
) -> NoReturn:
    target, transformed_features = preprocess_data
    model = train_model(transformed_features, target, training_params)
    check_is_fitted(model)
    assert isinstance(model, RandomForestClassifier)


def test_serialize_and_deserialize_model(
    training_params: RFParams,
    serialized_model_path: str,
    preprocess_data: Tuple[pd.Series, pd.DataFrame],
) -> NoReturn:
    target, transformed_features = preprocess_data
    model = train_model(transformed_features, target, training_params)
    serialize_model(model, serialized_model_path)
    assert os.path.exists(serialized_model_path)
    model = deserialize_model(serialized_model_path)
    assert isinstance(model, RandomForestClassifier)
    assert target.shape == model.predict(transformed_features).shape
