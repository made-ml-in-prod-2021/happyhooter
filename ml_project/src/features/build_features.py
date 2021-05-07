from typing import NoReturn

import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .custom_transformer import MyStandardScaler
from src.entities.feature_params import FeatureParams


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
            ),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", MyStandardScaler()),
        ]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target


def serialize_transformer(transformer: ColumnTransformer, output: str) -> NoReturn:
    with open(output, "wb") as f:
        pickle.dump(transformer, f)


def deserialize_transformer(input_: str) -> ColumnTransformer:
    with open(input_, "rb") as f:
        transformer = pickle.load(f)
    return transformer
