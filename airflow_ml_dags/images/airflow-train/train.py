import os
import pickle

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):
    data_path = os.path.join(input_dir, "train_data.csv")
    data_df = pd.read_csv(data_path)

    y_df = data_df[['target']]
    x_df = data_df.drop(['target'], axis=1)

    model = LogisticRegression()
    model.fit(x_df, y_df)

    os.makedirs(output_dir, exist_ok=True)
    output_model_path = os.path.join(output_dir, "model.pkl")
    with open(output_model_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()