from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pipeline_pdf_tasks import process_pdf, store_hse_data_to_mongo

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='hse_pdf_processing',
    default_args=default_args,
    description='Process HSE PDF reports and store in MongoDB',
    schedule_interval=None,  # Set to None for manual triggering; adjust as needed
    start_date=datetime(2025, 4, 17),
    catchup=False
) as dag:

    start = DummyOperator(task_id="start")

    # Configuration for each month's PDF
    month_configs = [
        {
            'currentMonth': 'Apr',
            'prevMonth': 'Mar',
            'fileName': 'health-services-employment-report-apr-2024.pdf',
            'pagenos': '57-94',
            'category': 'AcuteHospital'
        },
        {
            'currentMonth': 'May',
            'prevMonth': 'Apr',
            'fileName': 'health-services-employment-report-may-2024.pdf',
            'pagenos': '57-94',
            'category': 'AcuteHospital'
        },
        {
            'currentMonth': 'Jun',
            'prevMonth': 'May',
            'fileName': 'health-services-employment-report-june-2024.pdf',
            'pagenos': '57-94',
            'category': 'AcuteHospital'
        },
        {
            'currentMonth': 'Jul',
            'prevMonth': 'Jun',
            'fileName': 'health-services-employment-report-july-2024.pdf',
            'pagenos': '59-98',
            'category': 'AcuteHospital'
        }
    ]

    pdf_path = "/data"  # Update with actual path

    previous_task = start
    processed_data = []

    # Create sequential tasks for each PDF
    for config in month_configs:
        process_task = PythonOperator(
            task_id=f"process_{config['currentMonth'].lower()}_pdf",
            python_callable=process_pdf,
            op_kwargs={
                'month_config': config,
                'pdf_path': pdf_path,
                'processed_data': processed_data
            },
            provide_context=True
        )
        previous_task >> process_task
        previous_task = process_task

    store_task = PythonOperator(
        task_id="store_hse_data_to_mongo",
        python_callable=store_hse_data_to_mongo,
        op_kwargs={'processed_data': processed_data},
        provide_context=True
    )

    previous_task >> store_task