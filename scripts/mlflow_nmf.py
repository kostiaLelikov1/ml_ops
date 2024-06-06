import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from surprise import accuracy
from surprise.model_selection import cross_validate
from train_baseline_nmf import train

mlflow.set_tracking_uri("http://localhost:5785")
mlflow.set_experiment("Recomendation System")

with mlflow.start_run():
    model, trainset, testset, data = train(
        dataset_path="/datasets/ratings.csv", rating_scale=(0.5, 5.0)
    )

    predictions = model.test(testset)
    mlflow.log_param("n", len(data.df))
    mlflow.log_param("algorithm", "NMF")
    mlflow.log_param("dataset", f"MovieLens {len(data.df)}")
    mlflow.log_param("rating_scale", (0.5, 5.0))
    validation = cross_validate(
        model, data, measures=["RMSE", "MAE"], cv=5, verbose=True
    )

    rmse_values = validation["test_rmse"]
    mae_values = validation["test_mae"]

    fig, axs = plt.subplots(2, figsize=(10, 5))

    axs[0].plot(rmse_values)
    axs[0].set_title("RMSE values")
    axs[0].set_xlabel("Fold")
    axs[0].set_ylabel("RMSE")

    axs[1].plot(mae_values)
    axs[1].set_title("MAE values")
    axs[1].set_xlabel("Fold")
    axs[1].set_ylabel("MAE")

    plt.tight_layout()


    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", float(mae))

    mlflow.sklearn.log_model(
        model, f"model_NMF_MovieLens-{len(data.df)}_rating_scale-{(0.5, 5.0)}"
    )

    mlflow.log_artifact("/tmp/data.csv")
    mlflow.log_artifact("scripts/train_baseline_nmf.py")
