from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
# from scripts.parse_pdf import parse_pdf
from scripts.scrape_data import scrape_hse_data
# from scripts.preprocess_and_predict import preprocess_and_predict

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'hse_batch_pipeline',
    default_args=default_args,
    description='Daily scrape and PDF parse pipeline for HSE',
    schedule_interval='@daily',
    start_date=datetime(2025, 4, 16),
    catchup=False
) as dag:

    task_parse_pdf = PythonOperator(
        task_id='parse_pdf',
        python_callable=scrape_hse_data
    )

    task_scrape_data = PythonOperator(
        task_id='scrape_hse_data',
        python_callable=scrape_hse_data
    )

    task_preprocess_predict = PythonOperator(
        task_id='preprocess_and_predict',
        python_callable=scrape_hse_data
    )

    [task_parse_pdf, task_scrape_data] >> task_preprocess_predict
