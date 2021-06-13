import os
import pickle

import pandas as pd
import click


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as fin:
        model = pickle.load(fin)

    predicts = model.predict(data)

    os.makedirs(output_dir, exist_ok=True)
    predicts_path = os.path.join(output_dir, "predictions.csv")

    predicts_df = pd.DataFrame({"predictions": predicts})
    predicts_df.to_csv(predicts_path, index=False)


if __name__ == "__main__":
    predict()
