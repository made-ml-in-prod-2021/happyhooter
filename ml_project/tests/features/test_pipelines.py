from typing import NoReturn

import pytest
import pandas as pd
import numpy as np
from faker import Faker


from src.features import (
    build_categorical_pipeline,
    build_numerical_pipeline,
)


fake = Faker()
Faker.seed(42)


@pytest.fixture(scope="package")
def fake_categorical_data() -> pd.DataFrame:
    arr = np.array([fake.pyint(min_value=0, max_value=5) for _ in range(100)]).astype(
        float
    )
    mask = np.array([fake.pybool() for _ in range(100)])
    arr[mask] = np.nan
    data = {"cat_data": arr}
    return pd.DataFrame(data=data)


@pytest.fixture(scope="package")
def fake_numerical_data() -> pd.DataFrame:
    arr = np.array([fake.pyfloat(min_value=0, max_value=5) for _ in range(100)]).astype(
        float
    )
    mask = np.array([fake.pybool() for _ in range(100)])
    arr[mask] = np.nan
    data = {"num_data": arr}
    return pd.DataFrame(data=data)


def test_categorical_pipeline(fake_categorical_data: pd.DataFrame) -> NoReturn:
    expected_rows = fake_categorical_data.shape[0]
    expected_cols = 6
    cat_pipeline = build_categorical_pipeline()
    transformed_categorical_data = cat_pipeline.fit_transform(fake_categorical_data)
    transformed_categorical_data = pd.DataFrame(transformed_categorical_data.toarray())
    assert not pd.isnull(transformed_categorical_data).any().any()
    assert (expected_rows, expected_cols) == transformed_categorical_data.shape


def test_numerical_pipeline(fake_numerical_data: pd.DataFrame) -> NoReturn:
    num_pipeline = build_numerical_pipeline()
    transformed_numerical_data = num_pipeline.fit_transform(fake_numerical_data)
    transformed_numerical_data = pd.DataFrame(transformed_numerical_data)
    assert not pd.isnull(transformed_numerical_data).any().any()
    assert fake_numerical_data.shape == transformed_numerical_data.shape
