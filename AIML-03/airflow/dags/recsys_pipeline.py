"""Nightly retraining DAG: data -> Spark ETL -> train -> evaluate -> publish to Redis / Triton.

Mount this repo at /opt/recsys inside the Airflow image (see docker-compose.yml, profile "airflow").
The DAG was loaded with Airflow's DagBag (no import errors, 7 tasks) but not scheduled or run end to end
in the build sandbox.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2
    from airflow.operators.bash import BashOperator

REPO = "/opt/recsys"
CLI = f"cd {REPO} && python -m recsys.cli"

with DAG(
    dag_id="recsys_nightly",
    start_date=datetime(2026, 10, 20),
    schedule="0 2 * * *",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["recsys"],
) as dag:
    generate = BashOperator(task_id="ingest_or_generate", bash_command=f"{CLI} generate")
    etl = BashOperator(task_id="spark_etl", bash_command=f"{CLI} etl")
    retrieval = BashOperator(task_id="train_retrieval", bash_command=f"{CLI} train-retrieval")
    ranker = BashOperator(task_id="train_ranker", bash_command=f"{CLI} train-ranker")
    evaluate = BashOperator(task_id="evaluate", bash_command=f"{CLI} evaluate")
    export = BashOperator(task_id="export_to_triton", bash_command=f"{CLI} export-triton")
    # The API warms Redis from the new artifacts on start, so a rolling restart publishes the model.
    publish = BashOperator(task_id="reload_api", bash_command="curl -fsS -X POST $API_RELOAD_URL || true")

    generate >> etl >> retrieval >> ranker >> evaluate >> export >> publish
