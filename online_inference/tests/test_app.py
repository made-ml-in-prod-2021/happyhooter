import pandas as pd
from fastapi.testclient import TestClient

from app import app


def test_read_main(model_path, transformer_path):
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "it is entry point of our predictor"


def test_predict(dataset_path, model_path, transformer_path):
    with TestClient(app) as client:
        data = pd.read_csv(dataset_path)
        data["id"] = data.index + 1
        features = data.columns.tolist()
        data = data.values.tolist()[:5]
        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == 200
        assert response.json()[0]["disease"] == 1


def test_invalid_data_predict(model_path, transformer_path):
    with TestClient(app) as client:
        data = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
        features = data.columns.tolist()
        data = data.values.tolist()
        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == 400
