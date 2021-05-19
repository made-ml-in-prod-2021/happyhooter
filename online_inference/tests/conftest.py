import os

import pytest


@pytest.fixture(scope="session")
def dataset_path() -> str:
    return "data/raw/heart.csv"


@pytest.fixture(scope="session")
def model_path() -> str:
    os.environ['PATH_TO_MODEL'] = 'models/model.pkl'
    return os.getenv("PATH_TO_MODEL")


@pytest.fixture(scope="session")
def transformer_path() -> str:
    os.environ['PATH_TO_TRANSFORMER'] = 'models/transformer.pkl'
    return os.getenv("PATH_TO_TRANSFORMER")
