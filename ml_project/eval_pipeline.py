import os
import logging.config

import pandas as pd
from omegaconf import DictConfig
import hydra

from src.data import read_data
from src.entities.eval_pipeline_params import (
    EvaluatingPipelineParams,
    EvaluatingPipelineParamsSchema,
)
from src.features import make_features
from src.features.build_features import deserialize_transformer
from src.models import (
    predict_model,
    deserialize_model,
)


logger = logging.getLogger("ml_project/eval_pipeline")


def eval_pipeline(evaluating_pipeline_params: EvaluatingPipelineParams):
    logger.info("start evaluate pipeline")
    data = read_data(evaluating_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    logger.info("loading transformer")
    transformer = deserialize_transformer(evaluating_pipeline_params.pipeline_path)
    transformed_data = make_features(transformer, data)
    logger.info(f"transformed_data.shape is {transformed_data.shape}")

    logger.info("loading model")
    model = deserialize_model(evaluating_pipeline_params.model_path)

    logger.info("start prediction")
    predicts = predict_model(
        model,
        transformed_data,
    )
    logger.info(f"prediction.shape is {predicts.shape}")
    df_predicts = pd.DataFrame(predicts)

    df_predicts.to_csv(evaluating_pipeline_params.output_data_path, header=False)
    logger.info(
        f"predictions now in the file {evaluating_pipeline_params.output_data_path}"
    )
    return df_predicts


@hydra.main(config_path="configs", config_name="eval_config")
def eval_pipeline_command(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = EvaluatingPipelineParamsSchema()
    params = schema.load(cfg)
    eval_pipeline(params)


if __name__ == "__main__":
    eval_pipeline_command()
