<<<<<<< HEAD
import argparse
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.tensorflow

# =============================================================================
# Konfigurasi MLflow
# =============================================================================
# Pastikan server MLflow Anda berjalan di alamat dan port ini.
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Crypto_LSTM")

# =============================================================================
# Fungsi Helper
# =============================================================================

def load_preprocessed_data(preprocessed_dir: str):
    """
    Memuat data training dan testing yang sudah diproses dari file CSV.
    Fungsi ini juga memastikan konsistensi kolom dan tipe data.
    """
    X_train_path = os.path.join(preprocessed_dir, "X_train_preprocessed.csv")
    X_test_path = os.path.join(preprocessed_dir, "X_test_preprocessed.csv")

    if not os.path.exists(X_train_path) or not os.path.exists(X_test_path):
        raise FileNotFoundError(f"File CSV tidak ditemukan di direktori: {preprocessed_dir}")

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # Konversi ke numerik dan isi nilai yang hilang
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Sinkronisasi kolom untuk memastikan urutan dan jumlahnya sama
    common_cols = sorted(list(set(X_train.columns) & set(X_test.columns)))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    return X_train.values.astype(np.float32), X_test.values.astype(np.float32)

def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    """
    Mengubah bentuk array 2D [sampel, fitur] menjadi 3D [sampel, timesteps, fitur].
    Untuk model ini, kita menggunakan 1 timestep.
    """
    return np.reshape(X, (X.shape[0], 1, X.shape[1]))

def build_lstm_model(input_shape: tuple, output_dim: int) -> Sequential:
    """
    Membangun arsitektur model LSTM secara dinamis.
    
    Args:
        input_shape (tuple): Bentuk input untuk layer LSTM (timesteps, features).
        output_dim (int): Jumlah neuron di output layer, sesuai jumlah fitur target.
        
    Returns:
        Sequential: Model Keras yang sudah dikompilasi.
    """
    model = Sequential([
        # Layer LSTM pertama
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        # Layer LSTM kedua untuk menangkap pola yang lebih kompleks
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        # Layer Dense untuk interpretasi
        Dense(32, activation="relu"),
        # Layer output dengan jumlah neuron sesuai target prediksi
        Dense(output_dim)
    ])
    
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])
    return model

def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Mengevaluasi performa model pada data tes menggunakan MSE dan MAE.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae

# =============================================================================
# Fungsi Utama
# =============================================================================

def run_lstm_model(preprocessed_dir: str, out_dir: str, epochs: int, batch_size: int):
    """
    Orkestrasi utama untuk melatih, mengevaluasi, dan menyimpan model LSTM.
    """
    print("[INFO] Memulai proses LSTM Modeling...")
    print(f"[CONFIG] Direktori Preprocessed: {preprocessed_dir}")
    print(f"[CONFIG] Direktori Output: {out_dir}")
    print(f"[CONFIG] Epochs: {epochs}, Batch Size: {batch_size}")

    # 1. Memuat Data
    X_train, X_test = load_preprocessed_data(preprocessed_dir)
    # Model auto-regressive: input sama dengan target untuk prediksi satu langkah ke depan
    y_train, y_test = X_train, X_test
    
    print(f"[DATA] Shape X_train (sebelum reshape): {X_train.shape}")
    print(f"[DATA] Shape X_test (sebelum reshape): {X_test.shape}")

    # 2. Menyiapkan Data untuk LSTM
    X_train_lstm = reshape_for_lstm(X_train)
    X_test_lstm = reshape_for_lstm(X_test)
    
    print(f"[DATA] Shape X_train (setelah reshape): {X_train_lstm.shape}")
    print(f"[DATA] Shape X_test (setelah reshape): {X_test_lstm.shape}")

    # Mengaktifkan MLflow autologging untuk TensorFlow
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="LSTM-Crypto-Prediction"):
        # 3. Membangun Model Secara Dinamis
        # Bentuk input dan output ditentukan dari data, bukan hardcoded
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])  # (timesteps, features)
        output_dim = y_train.shape[1]  # Jumlah fitur yang akan diprediksi

        print(f"[MODEL] Membangun model dengan input_shape={input_shape} dan output_dim={output_dim}")
        model = build_lstm_model(input_shape, output_dim)
        model.summary()
        
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "input_shape": str(input_shape)})

        # 4. Melatih Model
        print("\n[TRAINING] Memulai training model LSTM...")
        model.fit(
            X_train_lstm, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_lstm, y_test),
            verbose=1,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )
        print("[TRAINING] Training selesai.")

        # 5. Mengevaluasi Model
        print("\n[EVALUATION] Mengevaluasi model pada data tes...")
        mse, mae = evaluate_model(model, X_test_lstm, y_test)
        print(f"[RESULT] Test MSE: {mse:.4f}")
        print(f"[RESULT] Test MAE: {mae:.4f}")

        # Logging metrik ke MLflow
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)

        # 6. Menyimpan Hasil
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "lstm_model.h5")
        model.save(model_path)
        print(f"[SAVE] Model berhasil disimpan di: {model_path}")

        report = {"test_mse": mse, "test_mae": mae}
        report_path = os.path.join(out_dir, "lstm_evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"[SAVE] Laporan evaluasi disimpan di: {report_path}")

    print("\n[INFO] Proses LSTM Modeling selesai!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM model for Crypto Price Prediction.")
    parser.add_argument("--preprocessed-dir", required=True, help="Direktori input berisi data yang sudah diproses (CSV).")
    parser.add_argument("--out-dir", required=True, help="Direktori output untuk menyimpan model dan laporan.")
    parser.add_argument("--epochs", type=int, default=50, help="Jumlah epochs untuk training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Ukuran batch untuk training.")
    args = parser.parse_args()
    
    run_lstm_model(args.preprocessed_dir, args.out_dir, args.epochs, args.batch_size)
=======
#!/usr/bin/env python3
"""
modelling.py
Baseline SKLearn models for single-step forecasting using processed_data.npz.
Uses mlflow.sklearn.autolog() to automatically log params/metrics/models.

Design:
 - clean functions, CLI options, logging
 - trains one RandomForestRegressor per feature (asset) predicting first horizon (y[:,0,f])
"""
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
>>>>>>> cdbbd87 (Initial commit - upload project SMSML)
