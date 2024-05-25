from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import boto3

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime.now(),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'tags': ['ml_ops']
}

dag = DAG(
    'upload_from_github_to_s3',
    default_args=default_args,
    description='Upload dataset from GitHub to S3',
    schedule_interval=timedelta(days=1),
)

GITHUB_URL = 'https://raw.githubusercontent.com/kostiaLelikov1/movielens-data/main/ratings.csv'
S3_BUCKET = 'my-ml-bucket'
S3_KEY = 'datasets/ratings.csv'

def download_and_upload_to_s3():
    # Download the file from GitHub
    response = requests.get(GITHUB_URL)
    response.raise_for_status()
    
    # Upload the file to S3
    s3 = boto3.client('s3', endpoint_url='http://s3.localhost:4569')
    s3.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=response.content)

upload_task = PythonOperator(
    task_id='download_and_upload_to_s3',
    python_callable=download_and_upload_to_s3,
    dag=dag,
)

upload_task
