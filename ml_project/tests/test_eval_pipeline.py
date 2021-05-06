import os
from typing import NoReturn

import pytest

from src.entities import (
    EvaluatingPipelineParams,
    TrainingPipelineParams,
)
from eval_pipeline import eval_pipeline
from train_pipeline import train_pipeline
from tests.test_full_pipeline import train_pipeline_params


@pytest.fixture(scope="package")
def eval_pipeline_params(
    dataset_path: str,
    serialized_model_path: str,
    output_predictions_path: str,
    transformer_path: str,
) -> EvaluatingPipelineParams:
    epp = EvaluatingPipelineParams(
        input_data_path=dataset_path,
        output_data_path=output_predictions_path,
        pipeline_path=transformer_path,
        model_path=serialized_model_path,
    )
    return epp


@pytest.fixture(scope="package")
def train_fake_data(train_pipeline_params: TrainingPipelineParams) -> NoReturn:
    train_pipeline(train_pipeline_params)


def test_eval_pipeline(
    eval_pipeline_params: EvaluatingPipelineParams, train_fake_data: NoReturn
) -> NoReturn:
    expected_rows = 100
    predictions = eval_pipeline(eval_pipeline_params)
    assert os.path.exists(eval_pipeline_params.output_data_path)
    assert expected_rows == predictions.shape[0]
    assert {0, 1} == set(predictions.iloc[:, 0])
