from typing import List

import pytest
import pandas as pd
from faker import Faker


NUM_ROWS = 100


@pytest.fixture(scope="session")
def dataset_path() -> str:
    return "tests/fake_data.csv"


@pytest.fixture(scope="session")
def output_predictions_path() -> str:
    return "tests/test_predictions.csv"


@pytest.fixture(scope="session")
def serialized_model_path() -> str:
    return "tests/test_model.pkl"


@pytest.fixture(scope="session")
def metric_path() -> str:
    return "tests/test_metrics.json"


@pytest.fixture(scope="session")
def transformer_path() -> str:
    return "tests/test_transformer.pkl"


@pytest.fixture(scope="session")
def target_col() -> str:
    return "target"


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope="session")
def fake_pd_dataframe() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(42)
    data = {
        "age": [fake.pyint(min_value=25, max_value=80) for _ in range(NUM_ROWS)],
        "sex": [fake.pyint(min_value=0, max_value=1) for _ in range(NUM_ROWS)],
        "cp": [fake.pyint(min_value=0, max_value=3) for _ in range(NUM_ROWS)],
        "trestbps": [fake.pyint(min_value=94, max_value=200) for _ in range(NUM_ROWS)],
        "chol": [fake.pyint(min_value=100, max_value=600) for _ in range(NUM_ROWS)],
        "fbs": [fake.pyint(min_value=0, max_value=1) for _ in range(NUM_ROWS)],
        "restecg": [fake.pyint(min_value=0, max_value=2) for _ in range(NUM_ROWS)],
        "thalach": [fake.pyint(min_value=70, max_value=205) for _ in range(NUM_ROWS)],
        "exang": [fake.pyint(min_value=0, max_value=1) for _ in range(NUM_ROWS)],
        "oldpeak": [fake.pyfloat(min_value=0, max_value=7) for _ in range(NUM_ROWS)],
        "slope": [fake.pyint(min_value=0, max_value=2) for _ in range(NUM_ROWS)],
        "ca": [fake.pyint(min_value=0, max_value=4) for _ in range(NUM_ROWS)],
        "thal": [fake.pyint(min_value=0, max_value=3) for _ in range(NUM_ROWS)],
        "target": [fake.pyint(min_value=0, max_value=1) for _ in range(NUM_ROWS)],
    }
    return pd.DataFrame(data=data)
