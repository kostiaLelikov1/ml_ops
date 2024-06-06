import pandas as pd
from surprise import Dataset, Reader, NMF
from surprise.model_selection import train_test_split
import boto3
from typing import Tuple


def train(
    dataset_path: str,
    rating_scale: Tuple[float, float] = (0.5, 5.0),
) -> NMF:
    s3 = boto3.client("s3", endpoint_url="http://s3.localhost:4569")
    s3.download_file("my-ml-bucket", dataset_path, "/tmp/data.csv")

    df = pd.read_csv("/tmp/data.csv")

    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

    model = NMF()
    model.fit(trainset)

    return model, trainset, testset, data
