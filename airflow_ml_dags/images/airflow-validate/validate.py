import os
import click
import pickle
import json

import pandas as pd
from sklearn.metrics import accuracy_score


@click.command("validate")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def validate(input_dir: str, model_dir: str, output_dir: str):
    val_path = os.path.join(input_dir, "val_data.csv")
    val_df = pd.read_csv(val_path)
    y_val_df = val_df[["target"]]
    x_val_df = val_df.drop(["target"], axis=1)

    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(x_val_df)
    scores = {"accuracy_score": accuracy_score(y_val_df.values, preds)}

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(scores, f)


if __name__ == '__main__':
    validate()