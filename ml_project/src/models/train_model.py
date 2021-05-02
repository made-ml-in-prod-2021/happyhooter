import pickle
from typing import Union, NoReturn

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.entities.train_params import LogRegParams, RFParams

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    train_params: Union[LogRegParams, RFParams],
) -> SklearnClassifierModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            criterion=train_params.criterion,
            random_state=train_params.random_state,
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            tol=train_params.tol,
            penalty=train_params.penalty,
            C=train_params.C,
            random_state=train_params.random_state,
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def serialize_model(model: SklearnClassifierModel, output: str) -> NoReturn:
    with open(output, "wb") as f:
        pickle.dump(model, f)


def deserialize_model(input_: str) -> SklearnClassifierModel:
    with open(input_, "rb") as fin:
        model = pickle.load(fin)
    return model
