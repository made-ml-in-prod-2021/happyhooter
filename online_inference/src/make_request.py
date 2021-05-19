import logging

import click
import numpy as np
import pandas as pd
import requests


logger = logging.getLogger("make_request")
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=8000)
@click.option("--dataset_path", default="data/raw/heart.csv")
def predict(host, port, dataset_path):
    data = pd.read_csv(dataset_path)
    data["id"] = data.index + 1
    request_features = list(data.columns)
    for i in range(data.shape[0]):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        logger.info(f"Request_data: {request_data}")
        response = requests.get(
            f"http://{host}:{port}/predict/",
            json={"data": [request_data], "features": request_features},
        )

        logger.info(
            f"Response status code: {response.status_code}, body json: {response.json()}"
        )


if __name__ == "__main__":
    predict()
