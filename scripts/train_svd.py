import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import joblib
import os
import boto3
import json
import time

def train_model():
    s3 = boto3.client('s3', endpoint_url='http://s3.localhost:4569')
    s3.download_file('my-ml-bucket', 'datasets/ratings.csv', '/tmp/ratings.csv')
    
    df = pd.read_csv('/tmp/ratings.csv')
    
    reader = Reader(rating_scale=(0.5, 5.0))

    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    
    model = SVD()
    model.fit(trainset)
    
    # Evaluate the model
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    timestamp = int(time.time())
    
    # Save the model
    if not os.path.exists('/models'):
        os.makedirs('/models')
    joblib.dump(model, f'/models/model-{timestamp}.pkl')
    
    # Save the metrics
    metrics = {
        'rmse': rmse,
        'mae': mae
    }
    if not os.path.exists('/metrics'):
        os.makedirs('/metrics')

    with open(f'/metrics/metrics-{timestamp}.json', 'w') as f:
        json.dump(metrics, f)
    
    s3.upload_file(f'/models/model-{timestamp}.pkl', 'my-ml-bucket', f'models/model-{timestamp}.pkl')
    s3.upload_file(f'/metrics/metrics-{timestamp}.json', 'my-ml-bucket', f'metrics/metrics-{timestamp}.json')

if __name__ == '__main__':
    train_model()
