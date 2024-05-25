from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import boto3
import json

app = FastAPI()

def load_model():
    s3 = boto3.client('s3', endpoint_url='http://s3.localhost:4569')
    sorted_models = sorted(s3.list_objects_v2(Bucket='my-ml-bucket', Prefix='models/')['Contents'], key=lambda x: x['LastModified'], reverse=True)
    if not sorted_models:
        raise Exception("No models found")
    latest_model = sorted_models[0]
    model_key = latest_model['Key']
    s3.download_file('my-ml-bucket', model_key, './model.pkl')
    return joblib.load('./model.pkl')

def load_metrics():
    s3 = boto3.client('s3', endpoint_url='http://s3.localhost:4569')
    sorted_metrics = sorted(s3.list_objects_v2(Bucket='my-ml-bucket', Prefix='metrics/')['Contents'], key=lambda x: x['LastModified'], reverse=True)
    if not sorted_metrics:
        raise Exception("No metrics found")
    latest_metrics = sorted_metrics[0]
    metrics_key = latest_metrics['Key']
    s3.download_file('my-ml-bucket', metrics_key, './metrics.json')
    with open('./metrics.json', 'r') as f:
        metrics = json.load(f)
    return metrics

model = load_model()

class PredictionRequest(BaseModel):
    user_id: int
    item_id: int

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict(request.user_id, request.item_id)
    return {"user_id": request.user_id, "item_id": request.item_id, "rating": prediction.est}

@app.get("/metrics")
def get_metrics():
    return metrics

@app.on_event("startup")
def startup_event():
    global model
    global metrics
    model = load_model()
    metrics = load_metrics()
