import argparse
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_lstm_model(preprocessed_dir, out_dir):
    # Load preprocessed data
    X_train = pd.read_csv(os.path.join(preprocessed_dir, r"D:\SMSML_Ratu_Chairunisa\Eksperimen_SML_Ratu-Chairunisa\preprocessing\crypto_preprocessing\X_train_preprocessed.csv")).values
    X_test = pd.read_csv(os.path.join(preprocessed_dir, r"D:\SMSML_Ratu_Chairunisa\Eksperimen_SML_Ratu-Chairunisa\preprocessing\crypto_preprocessing\X_test_preprocessed.csv")).values

    # Optionally, load y_train/y_test if available (for supervised learning)
    y_train_path = os.path.join(preprocessed_dir, "y_train.csv")
    y_test_path = os.path.join(preprocessed_dir, "y_test.csv")
    if os.path.exists(y_train_path) and os.path.exists(y_test_path):
        y_train = pd.read_csv(y_train_path).values
        y_test = pd.read_csv(y_test_path).values
    else:
        # For unsupervised or sequence prediction, use X as y (auto-regressive)
        y_train = X_train
        y_test = X_test

    # Here, time_steps=1 for simple LSTM, adjust as needed for sequence modeling
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(y_train.shape[1] if len(y_train.shape) > 1 else 1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    model.fit(X_train_lstm, y_train, epochs=30, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=1)

    # Predict and evaluate
    y_pred = model.predict(X_test_lstm)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[INFO] Test MSE: {mse:.4f}")
    print(f"[INFO] Test MAE: {mae:.4f}")

    # Save model
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "lstm_model.h5"))

    # Save evaluation report
    report = {
        "test_mse": mse,
        "test_mae": mae
    }
    import json
    with open(os.path.join(out_dir, "lstm_evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print("[INFO] LSTM Modeling selesai!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", required=True, help="Direktori hasil preprocessing")
    parser.add_argument("--out-dir", required=True, help="Direktori untuk simpan model")
    args = parser.parse_args()

    run_lstm_model(args.preprocessed_dir,