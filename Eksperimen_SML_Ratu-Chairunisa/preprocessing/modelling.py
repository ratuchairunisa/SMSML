import argparse
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

def run_modeling(preprocessed_dir, out_dir):
    # Load data hasil preprocessing
    X_train = pd.read_csv(os.path.join(preprocessed_dir, "D:\SMSML_Ratu_Chairunisa\Eksperimen_SML_Ratu-Chairunisa\preprocessing\crypto_preprocessing\X_train_preprocessed.csv"))
    X_test = pd.read_csv(os.path.join(preprocessed_dir, "D:\SMSML_Ratu_Chairunisa\Eksperimen_SML_Ratu-Chairunisa\preprocessing\crypto_preprocessing\X_test_preprocessed.csv"))

    # Load target
    y_train_path = os.path.join(preprocessed_dir, "y_train.csv")
    y_test_path = os.path.join(preprocessed_dir, "y_test.csv")

    if os.path.exists(y_train_path) and os.path.exists(y_test_path):
        y_train = pd.read_csv(y_train_path).squeeze()
        y_test = pd.read_csv(y_test_path).squeeze()
    else:
        raise ValueError("Target (y_train/y_test) tidak ditemukan. Pastikan --target-col dipakai di preprocessing.")

    # Training model Logistic Regression
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"[INFO] Accuracy: {acc:.4f}")

    # Simpan model
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.joblib"))

    # Simpan laporan evaluasi
    pd.DataFrame(report).transpose().to_csv(os.path.join(out_dir, "evaluation_report.csv"))

    print("[INFO] Modeling selesai!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", required=True, help="Direktori hasil preprocessing")
    parser.add_argument("--out-dir", required=True, help="Direktori untuk simpan model")
    args = parser.parse_args()

    run_modeling(args.preprocessed_dir, args.out_dir)
