from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pipeline_tasks import scrape_hse_data, store_hse_data_to_mongo

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='hse_batch_pipeline',
    default_args=default_args,
    description='Daily scrape → store HSE data in MongoDB',
    schedule_interval='@daily',
    start_date=datetime(2025, 4, 16),
    catchup=False
) as dag:

    start = DummyOperator(task_id="start")

    scrape_task = PythonOperator(
        task_id="scrape_hse_data",
        python_callable=scrape_hse_data,
        provide_context=True
    )

    store_task = PythonOperator(
        task_id="store_hse_data_to_mongo",
        python_callable=store_hse_data_to_mongo,
        provide_context=True
    )

    start >> scrape_task >> store_task
