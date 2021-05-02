from dataclasses import dataclass


@dataclass
class MainParams:
    input_data_path: str
    output_model_path: str
    output_transformer_path: str
    metric_path: str
