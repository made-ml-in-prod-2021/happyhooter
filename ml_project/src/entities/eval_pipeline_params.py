from dataclasses import dataclass

from marshmallow_dataclass import class_schema


@dataclass()
class EvaluatingPipelineParams:
    input_data_path: str
    output_data_path: str
    pipeline_path: str
    model_path: str


EvaluatingPipelineParamsSchema = class_schema(EvaluatingPipelineParams)
