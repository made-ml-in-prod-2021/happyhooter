ml_project
==============================

First homework

Project Organization
------------

    ├── configs
    │   └── feature_params <- Configs for categorical/numerical features
    │   └── main           <- Configs for data/models/etc. paths
    │   └── splitting_params <- Configs for splitting parametres
    │   └── train_params   <- Configs for different models parametres
    │   └── config.yaml    <- Default config for hydra
    │   └── eval_config.yaml <- Evaluating config for hydra 
    │
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.                       
    │
    ├── outputs            <- Hydra logs
    │
    ├── src                <- Source code for use in this project.
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── build_features.py
    │   │   └── custom_transformer.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── entities       <- Dataclasses for different parametres
    │       └── eval_pipeline_params.py
    │       └── feature_params.py
    │       └── main_params.py
    │       └── split_params.py
    │       └── train_params.py
    │       └── train_pipeline_params.py
    │
    ├── tests              <- Tests for project
    ├── eval_pipeline.py   <- pipeline for making predictions
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    └── train_pipeline.py  <- pipeline for training model on raw data


--------

Обучение модели
------------

Необходимы следующие конфиги для обучения модели:

    ├── configs
    │   └── feature_params
    │   │   └── all_features.yaml <- Конфиг, где описываются категориальные/численные/таргет фичи
    │   │
    │   ├── main           
    │   │   └── main_config.yaml <- Конфиг с путями до данных/моделей/метрик и т.д.
    │   │
    │   ├── splitting_params
    │   │   └── splitting_params.yaml <- Конфиг с параметрами для split
    │   │
    │   ├── train_params
    │   │   └── rf.yaml <- Конфиг с параметрами модели классификации
    │   │
    │   ├── config.yaml    <- Главный конфиг с defaults для hydra

Для запуска обучения необходимо перейти в папку `ml_project` и в командной строке написать `python train_pipeline.py`

--------

Предсказания модели
------------

Необходимы следующие конфиги для предсказания модели:

    ├── configs
        └── eval_config.yaml <- Конфиг, где описываются пути до данных, модели и т.д.
   

Для запуска предсказания необходимо перейти в папку `ml_project` и в командной строке написать `python eval_pipeline.py`

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
