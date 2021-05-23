import logging
import os
import pickle
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from src.entities import HeartDiseaseModel, DiseaseResponse


logger = logging.getLogger(__name__)


def load_transformer(path: str) -> ColumnTransformer:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(path: str) -> LogisticRegression:
    with open(path, "rb") as f:
        return pickle.load(f)


model: Optional[LogisticRegression] = None
transformer: Optional[ColumnTransformer] = None


def make_predict(
        data: List,
        features: List[str],
        model: LogisticRegression,
        transformer: ColumnTransformer,
) -> List[DiseaseResponse]:
    try:
        data = pd.DataFrame(data, columns=features)
        ids = [int(x) for x in data["id"]]
        data.drop(["id"], axis=1, inplace=True)
        transformed_data = pd.DataFrame(transformer.transform(data))
        predicts = model.predict(transformed_data)

        return [
            DiseaseResponse(id=id_, disease=disease)
            for id_, disease in zip(ids, predicts)
        ]
    except:
        raise HTTPException(status_code=400)


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model_and_transformer():
    global model
    global transformer
    model_path = os.getenv("PATH_TO_MODEL")
    transformer_path = os.getenv("PATH_TO_TRANSFORMER")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    if transformer_path is None:
        err = f"PATH_TO_TRANSFORMER {transformer_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_model(model_path)
    transformer = load_transformer(transformer_path)


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=List[DiseaseResponse])
def predict(request: HeartDiseaseModel):
    return make_predict(request.data, request.features, model, transformer)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
