from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3
import subprocess

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime.now(),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'tags': ['ml_ops']
}

S3_BUCKET = 'my-ml-bucket'
S3_MODELS_PREFIX = 'models/'

dag = DAG(
    'train_and_deploy_model',
    default_args=default_args,
    description='Train and deploy model on data update',
    schedule_interval=None,
)

def train_model():
    result = subprocess.run(['python', '/scripts/train_svd.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Training script failed: {result.stderr}")
    print(result.stdout)

def manage_models():
    s3 = boto3.client('s3', endpoint_url='http://s3.localhost:4569')
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_MODELS_PREFIX)

    if 'Contents' not in response:
        return

    all_models = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
    
    if len(all_models) > 3:
        for model in all_models[3:]:
            s3.delete_object(Bucket=S3_BUCKET, Key=model['Key'])


train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

manage_models_task = PythonOperator(
    task_id='manage_models',
    python_callable=manage_models,
    dag=dag,
)

train_task >> manage_models_task 
