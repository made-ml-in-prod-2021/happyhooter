import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "download_dag",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1),
) as dag:
    download = DockerOperator(
        image="happyhooter/airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=["/Users/happyhooter/Desktop/MADE/semester_2/ML_in_prod/airflow_ml_dags/data:/data"],
    )

    download
