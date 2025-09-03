import argparse
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.tensorflow
import joblib

# =============================================================================
# Konfigurasi MLflow
# =============================================================================
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Crypto_LSTM_Forecasting")

# =============================================================================
# Fungsi-Fungsi Helper
# =============================================================================

def load_preprocessed_data(preprocessed_dir: str):
    """Memuat data training dan testing yang sudah diproses dari file CSV."""
    X_train_path = os.path.join(preprocessed_dir, "X_train_preprocessed.csv")
    X_test_path = os.path.join(preprocessed_dir, "X_test_preprocessed.csv")

    if not os.path.exists(X_train_path) or not os.path.exists(X_test_path):
        raise FileNotFoundError(f"File CSV tidak ditemukan di direktori: {preprocessed_dir}")

    X_train_df = pd.read_csv(X_train_path)
    X_test_df = pd.read_csv(X_test_path)

    # Simpan nama kolom untuk inversi nanti
    columns = X_train_df.columns

    X_train = X_train_df.apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    X_test = X_test_df.apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    
    return X_train, X_test, columns

def load_scaler(scaler_path: str) -> MinMaxScaler:
    """Memuat scaler yang telah disimpan."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"File scaler tidak ditemukan di: {scaler_path}. "
                                "Scaler ini penting untuk mengembalikan prediksi ke skala aslinya.")
    return joblib.load(scaler_path)

def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    """Mengubah bentuk array 2D menjadi 3D untuk input LSTM (timesteps = 1)."""
    return np.reshape(X, (X.shape[0], 1, X.shape[1]))

def build_lstm_model(input_shape: tuple, output_dim: int) -> Sequential:
    """Membangun arsitektur model LSTM secara dinamis."""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(output_dim)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])
    return model

def predict_future(model: Sequential, last_input: np.ndarray, n_steps: int, scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Melakukan prediksi iteratif untuk n_steps ke depan.
    """
    print(f"\n[FORECASTING] Memulai prediksi untuk {n_steps} hari ke depan...")
    future_predictions_scaled = []
    current_input = last_input.copy()

    for i in range(n_steps):
        # Prediksi satu langkah ke depan
        pred_scaled = model.predict(current_input, verbose=0)
        future_predictions_scaled.append(pred_scaled[0])
        
        # Perbarui input untuk prediksi berikutnya dengan hasil prediksi saat ini
        current_input = np.reshape(pred_scaled, (1, 1, pred_scaled.shape[1]))
        print(f"  - Hari ke-{i+1} diprediksi.")

    # Kembalikan prediksi ke skala aslinya
    future_predictions_unscaled = scaler.inverse_transform(np.array(future_predictions_scaled))
    print("[FORECASTING] Prediksi selesai dan sudah dikembalikan ke skala semula.")
    return future_predictions_unscaled


# =============================================================================
# Fungsi Utama
# =============================================================================

def run_lstm_pipeline(preprocessed_dir: str, out_dir: str, epochs: int, batch_size: int, prediction_days: int):
    """
    Orkestrasi utama untuk melatih, mengevaluasi, menyimpan, dan melakukan forecasting.
    """
    print("[INFO] Memulai pipeline LSTM Modeling & Forecasting...")
    
    # 1. Memuat Data dan Scaler
    X_train, X_test, columns = load_preprocessed_data(preprocessed_dir)
    scaler_path = os.path.join(preprocessed_dir, 'scaler.joblib')
    scaler = load_scaler(scaler_path)
    y_train, y_test = X_train, X_test
    
    # 2. Menyiapkan Data untuk LSTM
    X_train_lstm = reshape_for_lstm(X_train)
    X_test_lstm = reshape_for_lstm(X_test)

    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="LSTM-Training-and-Forecasting"):
        # 3. Membangun Model Secara Dinamis (FIX untuk error dimensi)
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        output_dim = y_train.shape[1]
        
        print(f"[MODEL] Membangun model dengan input_shape={input_shape} dan output_dim={output_dim}")
        model = build_lstm_model(input_shape, output_dim)
        model.summary()

        # 4. Melatih Model
        print("\n[TRAINING] Memulai training model LSTM...")
        model.fit(
            X_train_lstm, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(X_test_lstm, y_test),
            verbose=1,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        )

        # 5. Mengevaluasi Model
        print("\n[EVALUATION] Mengevaluasi model pada data tes...")
        y_pred = model.predict(X_test_lstm)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"[RESULT] Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

        # 6. Menyimpan Hasil
        os.makedirs(out_dir, exist_ok=True)
        model.save(os.path.join(out_dir, "lstm_model.h5"))
        report = {"test_mse": mse, "test_mae": mae}
        with open(os.path.join(out_dir, "lstm_evaluation_report.json"), "w") as f:
            json.dump(report, f, indent=4)
        
        # 7. Melakukan Forecasting
        last_known_input = np.reshape(X_test[-1], (1, 1, X_test.shape[1]))
        future_preds_unscaled = predict_future(model, last_known_input, prediction_days, scaler)
        
        # Simpan hasil prediksi
        forecast_df = pd.DataFrame(future_preds_unscaled, columns=columns)
        forecast_df.index = pd.to_datetime(pd.to_datetime('today').date()) + pd.to_timedelta(np.arange(1, prediction_days + 1), 'D')
        forecast_path = os.path.join(out_dir, f"forecast_{prediction_days}_days.csv")
        forecast_df.to_csv(forecast_path)
        print(f"\n[SAVE] Prediksi {prediction_days} hari ke depan disimpan di: {forecast_path}")

    print("\n[INFO] Pipeline LSTM selesai!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model and forecast future crypto prices.")
    parser.add_argument("--preprocessed-dir", required=True, help="Direktori berisi data dan scaler yang sudah diproses.")
    parser.add_argument("--out-dir", required=True, help="Direktori output untuk menyimpan model dan laporan.")
    parser.add_argument("--epochs", type=int, default=50, help="Jumlah epochs untuk training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Ukuran batch untuk training.")
    parser.add_argument("--prediction-days", type=int, default=30, help="Jumlah hari ke depan untuk diprediksi.")
    args = parser.parse_args()
    
    run_lstm_pipeline(
        args.preprocessed_dir, 
        args.out_dir, 
        args.epochs, 
        args.batch_size, 
        args.prediction_days
    )
