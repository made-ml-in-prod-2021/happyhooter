import os
import json
import logging.config
from typing import Dict

from omegaconf import DictConfig
import hydra


from src.data import read_data, split_train_val_data
from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    TrainingPipelineParamsSchema,
)
from src.features import make_features
from src.features.build_features import (
    extract_target,
    build_transformer,
    serialize_transformer,
)
from src.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)


logger = logging.getLogger("ml_project/train_pipeline")


def train_pipeline(
    training_pipeline_params: TrainingPipelineParams,
) -> Dict[str, float]:
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.main.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")
    logger.info(f"train_df.type is {type(train_df)}")

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    serialize_transformer(
        transformer, training_pipeline_params.main.output_transformer_path
    )
    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    val_features = make_features(transformer, val_df)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)

    logger.info(f"val_features.shape is {val_features.shape}")
    predicts = predict_model(
        model,
        val_features,
    )

    metrics = evaluate_model(predicts, val_target)

    with open(training_pipeline_params.main.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    serialize_model(model, training_pipeline_params.main.output_model_path)

    return metrics


@hydra.main(config_path="configs", config_name="config")
def train_pipeline_command(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
