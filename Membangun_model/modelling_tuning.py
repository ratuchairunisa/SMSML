import argparse
from pathlib import Path
import numpy as np
import mlflow
import dagshub
import os
import logging
import json
import sklearn

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error

mlflow.set_experiment("lstm_tuning")

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def load_data(npz_path: Path):
    d = np.load(npz_path)
    return d['X_train'], d['Y_train'], d['X_val'], d['Y_val'], d['X_test'], d['Y_test']

def build_lstm(n_past, n_features, n_future, units=64, dropout=0.2, lr=1e-3):
    model = keras.Sequential([
        layers.Input(shape=(n_past, n_features)),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(units//2, activation='relu'),
        layers.Dense(n_future * n_features),
        layers.Reshape((n_future, n_features))
    ])
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    return model

def mape(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def directional_accuracy(y_true, y_pred):
    true_diff = np.sign(y_true[:,1:] - y_true[:,:-1])
    pred_diff = np.sign(y_pred[:,1:] - y_pred[:,:-1])
    match = (true_diff == pred_diff)
    return float(np.mean(match))

def evaluate_multi(y_true, y_pred):
    mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    rmse = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1), squared=False)
    mapep = mape(y_true, y_pred)
    dacc = directional_accuracy(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mapep), "directional_acc": float(dacc)}

def plot_history(history, out_path: Path):
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend()
    plt.xlabel('epoch'); plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def run_grid(args):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(Path(args.input))
    n_past = X_train.shape[1]; n_features = X_train.shape[2]; n_future = Y_train.shape[1]
    logging.info(f"Shapes: X_train {X_train.shape} Y_train {Y_train.shape}")

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        logging.info(f"MLflow tracking uri: {args.tracking_uri}")

    grid = {
        "units":[32, 64],
        "lr":[1e-3, 1e-4],
        "batch_size":[32]
    }

    import itertools
    combos = list(itertools.product(grid['units'], grid['lr'], grid['batch_size']))

    for units, lr, batch_size in combos:
        params = {"units": units, "lr": float(lr), "batch_size": int(batch_size)}
        run_name = f"lstm_u{units}_lr{lr}_b{batch_size}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)

            model = build_lstm(n_past, n_features, n_future, units=units, dropout=0.2, lr=lr)
            callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
            ]

            history = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=50,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )

            # predictions
            preds = model.predict(X_test)
            metrics = evaluate_multi(Y_test, preds)

            # manual logging of metrics
            for k,v in metrics.items():
                mlflow.log_metric(k, v)

            # save model locally then log artifact
            out_dir = Path("tmp_runs") / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            model_path = str(out_dir / "keras_model.keras")
            model.save(model_path)
            mlflow.log_artifacts(str(out_dir), artifact_path="model_artifacts")

            # plot loss and log as artifact
            plot_path = out_dir / "loss.png"
            plot_history(history, plot_path)
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

            # also save metadata
            meta = {"params": params, "metrics": metrics}
            with open(out_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            mlflow.log_artifact(str(out_dir / "meta.json"), artifact_path="meta")

            logging.info(f"Run {run_name} logged: {metrics}")

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="Membangun_model/processed/processed_data.npz")
    parser.add_argument("--tracking_uri", type=str, default=None)
    args = parser.parse_args()
    run_grid(args)
