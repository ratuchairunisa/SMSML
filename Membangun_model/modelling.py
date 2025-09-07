import argparse
import logging
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import dagshub
import mlflow
import mlflow.sklearn
import os

from dagshub import init
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

mlflow.set_experiment("rf_baseline_multistep")

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", force=True)

npz_path = Path("D:/smsml_ratu-chairunisa/Membangun_model/processed/processed_data.npz")

def load_processed(npz_path: Path):
    data = np.load(npz_path)
    X_train = data['X_train']; Y_train = data['Y_train']
    X_val = data['X_val']; Y_val = data['Y_val']
    X_test = data['X_test']; Y_test = data['Y_test']
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def flatten_for_sklearn(X):
    # X shape: (n_samples, n_past, n_features) -> (n_samples, n_past * n_features)
    n = X.shape[0]
    return X.reshape(n, -1)

def evaluate_and_log(y_true, y_pred, prefix=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": float(mae), "rmse": float(rmse)}

def train_baseline(args):
    npz = Path(args.input)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_processed(npz)

    # flatten feature windows for sklearn baseline
    Xtr = flatten_for_sklearn(X_train)
    Xte = flatten_for_sklearn(X_test)

    n_features = X_train.shape[2]  # number of assets
    logging.info(f"Data shapes: Xtr {Xtr.shape}, Y_train {Y_train.shape}, n_features {n_features}")

    # set tracking uri if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        logging.info(f"MLflow tracking URI set to {args.tracking_uri}")

    # enable autolog for sklearn
    mlflow.sklearn.autolog()

    feature_names = [f"asset_{i}" for i in range(n_features)]

    for fid in range(n_features):
        # target = first future day for feature fid
        ytr = Y_train[:, :, fid]   # shape: (n_samples, n_future)
        yte = Y_test[:, :, fid]

        run_name = f"mor_asset_{fid}"
        with mlflow.start_run(run_name=run_name):
            # model (default hyperparams) - no hyperparameter tuning
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=args.n_estimators, random_state=42, n_jobs=-1))
            model.fit(Xtr, ytr)

            preds = model.predict(Xte)
            metrics = evaluate_and_log(yte, preds)

            # manual log of metrics too (autolog already logs some, but explicit is fine)
            mlflow.log_metric("test_mae", metrics["mae"])
            mlflow.log_metric("test_rmse", metrics["rmse"])

            logging.info(f"Asset {fid}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="processed/processed_data.npz")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--tracking_uri", type=str, default=None, help="If set, mlflow.set_tracking_uri(...)")
    args = parser.parse_args()
    train_baseline(args)
