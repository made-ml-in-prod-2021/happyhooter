import os
from typing import List

import pytest
import pandas as pd

from src.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    LogRegParams,
)
from src.entities.main_params import MainParams
from train_pipeline import train_pipeline


@pytest.fixture
def train_pipeline_params(
    fake_pd_dataframe: pd.DataFrame,
    dataset_path: str,
    serialized_model_path: str,
    metric_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    transformer_path: str,
) -> TrainingPipelineParams:
    fake_pd_dataframe.to_csv(dataset_path, index=False)
    tpp = TrainingPipelineParams(
        main=MainParams(
            input_data_path=dataset_path,
            metric_path=metric_path,
            output_model_path=serialized_model_path,
            output_transformer_path=transformer_path,
        ),
        splitting_params=SplittingParams(val_size=0.1, random_state=42),
        feature_params=FeatureParams(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            target_col=target_col,
        ),
        train_params=LogRegParams(),
    )
    return tpp


def test_full_pipeline(train_pipeline_params: TrainingPipelineParams):
    metrics = train_pipeline(train_pipeline_params)
    assert 0 < metrics["roc_auc_score"] <= 1
    assert 0 < metrics["accuracy_score"] <= 1
    assert 0 < metrics["f1_score"] <= 1
    assert os.path.exists("tests/test_metrics.json")
    assert os.path.exists("tests/test_model.pkl")
