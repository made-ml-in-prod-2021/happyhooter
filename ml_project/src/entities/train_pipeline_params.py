from typing import Union

from dataclasses import dataclass
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import LogRegParams, RFParams
from .main_params import MainParams
from marshmallow_dataclass import class_schema


@dataclass()
class TrainingPipelineParams:
    main: MainParams
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: Union[LogRegParams, RFParams]


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)
