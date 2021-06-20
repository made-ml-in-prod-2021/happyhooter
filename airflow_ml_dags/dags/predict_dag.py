import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "predict_dag",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1),
) as dag:
    wait_for_data = FileSensor(
        task_id="wait-for-data",
        poke_interval=5,
        retries=5,
        filepath="data/raw/{{ ds }}/data.csv",
    )
    wait_for_model = FileSensor(
        task_id="wait-for-model",
        poke_interval=5,
        retries=5,
        filepath="data/models/{{ ds }}/model.pkl",
    )
    predict = DockerOperator(
        image="happyhooter/airflow-predict",
        command="--input-dir /data/raw/{{ ds }} "
        "--output-dir /data/predictions/{{ ds }} "
        "--model-dir data/models/{{ ds }} ",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=["/Users/happyhooter/Desktop/MADE/semester_2/ML_in_prod/airflow_ml_dags/data:/data"],
    )

    [wait_for_data, wait_for_model] >> predict
