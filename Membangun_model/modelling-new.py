import argparse
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import mlflow
import mlflow.tensorflow

def run_lstm_model(preprocessed_dir, out_dir):
    print("[INFO] Memulai LSTM Modeling")
    print("[INFO] Preprocessed directory:", preprocessed_dir)
    print("[INFO] Output directory:", out_dir)

    # Load preprocessed data
    X_train_path = os.path.join(preprocessed_dir, "X_train_preprocessed.csv")
    X_test_path  = os.path.join(preprocessed_dir, "X_test_preprocessed.csv")

    if not os.path.exists(X_train_path) or not os.path.exists(X_test_path):
        raise FileNotFoundError("File preprocessed CSV tidak ditemukan di folder yang diberikan.")

    X_train = pd.read_csv(X_train_path).values
    X_test  = pd.read_csv(X_test_path).values

    # Untuk LSTM prediksi time series, gunakan X sebagai target auto-regressive
    y_train = X_train
    y_test  = X_test

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_test shape: {X_test.shape}")

    # Reshape data untuk LSTM (samples, timesteps=1, features)
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm  = np.reshape(X_test,  (X_test.shape[0], 1, X_test.shape[1]))

    print(f"[INFO] X_train reshaped for LSTM: {X_train_lstm.shape}")
    print(f"[INFO] X_test reshaped for LSTM: {X_test_lstm.shape}")

    # MLflow autolog
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="LSTM-Model"):
        # Build LSTM model
        model = Sequential([
            LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(y_train.shape[1] if len(y_train.shape) > 1 else 1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("[INFO] LSTM model built successfully")
        model.summary()

        # Train model
        print("[INFO] Memulai training LSTM...")
        history = model.fit(
            X_train_lstm, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_test_lstm, y_test),
            verbose=1
        )
        print("[INFO] Training selesai")

        # Predict and evaluate
        y_pred = model.predict(X_test_lstm)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"[INFO] Test MSE: {mse:.4f}")
        print(f"[INFO] Test MAE: {mae:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)

        # Save model
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "lstm_model.h5")
        model.save(model_path)
        print(f"[INFO] Model tersimpan di: {model_path}")

        # Save evaluation report
        report_path = os.path.join(out_dir, "lstm_evaluation_report.json")
        report = {"test_mse": mse, "test_mae": mae}
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"[INFO] Evaluation report tersimpan di: {report_path}")

        print("[INFO] LSTM Modeling selesai!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", required=True, help="Direktori hasil preprocessing")
    parser.add_argument("--out-dir", required=True, help="Direktori untuk simpan model")
    args = parser.parse_args()

    run_lstm_model(args.preprocessed_dir, args.out_dir)
