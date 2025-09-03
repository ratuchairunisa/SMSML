import mlflow
import mlflow.tensorflow

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Tentukan path dan direktori output
raw_path = r"D:\SMSML_Ratu_Chairunisa\Eksperimen_SML_Ratu-Chairunisa\crypto.csv"
out_dir = r"D:\SMSML_Ratu_Chairunisa\Eksperimen_SML_Ratu-Chairunisa\preprocessing\crypto_preprocessing"

def run_preprocessing(raw_path, out_dir):
    # Load dataset
    df = pd.read_csv(raw_path)
    print(f"[INFO] Loaded dataset: {df.shape}")

    # Pastikan kolom Date ada
    if "Date" not in df.columns:
        raise ValueError("Kolom 'Date' tidak ditemukan dalam dataset!")

    # Ubah kolom Date ke datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Ekstrak fitur numerik dari Date
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek

    # Drop kolom Date asli
    df = df.drop(columns=["Date"])

    # Train-test split
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)

    # Pilih hanya kolom numerik
    X_train = X_train.select_dtypes(include=np.number)
    X_test = X_test.select_dtypes(include=np.number)

    # Pipeline preprocessing (scaling)
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Fit-transform train, transform test
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # Konversi kembali ke DataFrame
    X_train_processed = pd.DataFrame(X_train_processed, columns=X_train.columns)
    X_test_processed = pd.DataFrame(X_test_processed, columns=X_test.columns)

    # Buat output directory
    os.makedirs(out_dir, exist_ok=True)

    # Simpan pipeline
    joblib.dump(pipeline, os.path.join(out_dir, "preprocess_pipeline.joblib"))

    # Simpan hasil preprocess
    X_train_processed.to_csv(os.path.join(out_dir, "X_train_preprocessed.csv"), index=False)
    X_test_processed.to_csv(os.path.join(out_dir, "X_test_preprocessed.csv"), index=False)

    # Simpan laporan preprocessing
    report = {
        "raw_shape": df.shape,
        "train_shape": X_train_processed.shape,
        "test_shape": X_test_processed.shape,
        "features": list(X_train.columns),
        "scaler": "StandardScaler"
    }
    with open(os.path.join(out_dir, "preprocessing_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print("[INFO] Preprocessing selesai! File tersimpan di:", out_dir)


if __name__ == "__main__":
    run_preprocessing(raw_path, out_dir)
