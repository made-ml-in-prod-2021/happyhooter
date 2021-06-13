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
    "train_dag",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(1),
) as dag:
    wait_for_data = FileSensor(
        task_id="wait-for-data",
        poke_interval=5,
        retries=5,
        filepath="data/raw/{{ ds }}/data.csv",
    )
    wait_for_target = FileSensor(
        task_id="wait-for-target",
        poke_interval=5,
        retries=5,
        filepath="data/raw/{{ ds }}/target.csv",
    )
    preprocess = DockerOperator(
        image="happyhooter/airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=["/Users/happyhooter/Desktop/MADE/semester_2/ML_in_prod/airflow_ml_dags/data:/data"],
    )

    split = DockerOperator(
        image="happyhooter/airflow-split",
        command="--input-dir /data/processed/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=["/Users/happyhooter/Desktop/MADE/semester_2/ML_in_prod/airflow_ml_dags/data:/data"],
    )

    train = DockerOperator(
        image="happyhooter/airflow-train",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=["/Users/happyhooter/Desktop/MADE/semester_2/ML_in_prod/airflow_ml_dags/data:/data"],
    )

    validate = DockerOperator(
        image="happyhooter/airflow-validate",
        command="--input-dir /data/processed/{{ ds }} \
         --model-dir /data/models/{{ ds }} --output-dir /data/metrics/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=["/Users/happyhooter/Desktop/MADE/semester_2/ML_in_prod/airflow_ml_dags/data:/data"],
    )

    [wait_for_data, wait_for_target] >> preprocess >> split >> train >> validate
