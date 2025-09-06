#!/usr/bin/env python3
"""
modelling.py
Simple LSTM-ish training script that:
 - loads processed_data.npz (X_train, Y_train, X_val, Y_val, X_test, Y_test)
 - trains a small keras model (configurable)
 - logs run & model to MLflow (mlflow.keras.log_model(..., artifact_path="model"))
 - saves local artifacts to 'output' folder and writes run_id to output/run_id.txt
"""

import argparse
from pathlib import Path
import json
import numpy as np
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os

def load_npz(p):
    d = np.load(p)
    return d['X_train'], d['Y_train'], d['X_val'], d['Y_val'], d['X_test'], d['Y_test']

def build_model(n_past, n_features, n_future, units=64):
    m = models.Sequential([
        layers.Input(shape=(n_past, n_features)),
        layers.LSTM(units),
        layers.Dense(units//2, activation='relu'),
        layers.Dense(n_future * n_features),
        layers.Reshape((n_future, n_features))
    ])
    m.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return m

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    rmse = math.sqrt(((y_true.reshape(-1)-y_pred.reshape(-1))**2).mean())
    return {"mae": float(mae), "rmse": float(rmse)}

def main(args):
    p = Path(args.input)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_npz(p)

    n_past = X_train.shape[1]; 
    n_features = X_train.shape[2]; 
    n_future = Y_train.shape[1]

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    # make output dir
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("n_past", args.n_past)
        mlflow.log_param("n_future", args.n_future)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)

        model = build_model(n_past, n_features, n_future, units=args.units)
        # quick train (epochs configurable)
        model.fit(X_train, Y_train,
                  validation_data=(X_val, Y_val),
                  epochs=args.epochs, batch_size=args.batch_size, verbose=1)

        # evaluate
        preds = model.predict(X_test)
        metrics = eval_metrics(Y_test, preds)
        for k,v in metrics.items():
            mlflow.log_metric(k, v)

        # log model to MLflow with artifact_path 'model' (so runs:/<run_id>/model exists)
        mlflow.keras.log_model(model, artifact_path="model")

        # also save local copy to artifacts folder for GH Actions upload
        local_model_dir = out / "keras_model.keras"
        model.save(str(local_model_dir))

        # write run id file for workflow consumption
        with open(out / "run_id.txt", "w") as f:
            f.write(run_id)

        print(f"MLflow run_id: {run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n_past", type=int, default=30)
    parser.add_argument("--n_future", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--output", type=str, default="artifacts")
    parser.add_argument("--tracking_uri", type=str, default=None)
    args = parser.parse_args()
    main(args)
