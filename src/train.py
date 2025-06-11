import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression

from src.evaluate import main as evaluate_main

if __name__ == "__main__":
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("assignment-3-mlflow")
    with mlflow.start_run(run_name="train") as run:
        train_dataset = pd.read_csv("data/train.csv")
        y: np.ndarray = train_dataset.loc[:, "target"].values.astype("float32")
        X: np.ndarray = train_dataset.drop("target", axis=1).values
        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=100)
        clf.fit(X, y)
        train_score = clf.score(X, y)
        mlflow.log_metric("train_accuracy", train_score)
        signature = infer_signature(X, clf.predict(X))
        mlflow.sklearn.log_model(
            clf,
            "model",
            signature=signature,
            registered_model_name="assignment-3-mlflow-model",
        )
        joblib.dump(clf, "models/model.joblib")
        with mlflow.start_run(run_name="evaluate", nested=True):
            evaluate_main()
