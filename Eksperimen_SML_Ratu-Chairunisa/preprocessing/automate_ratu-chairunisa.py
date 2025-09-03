import os
import pandas as pd
import numpy as np

csv_path = r"D:\SMSML_Ratu_Chairunisa\Eksperimen_SML_Ratu-Chairunisa\crypto.csv"
out_dir = r"D:\SMSML_Ratu_Chairunisa\Eksperimen_SML_Ratu-Chairunisa\preprocessing\crypto_preprocessing"

def load_dataset(csv_path):
    """Load dataset and set Date as datetime index."""
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def check_missing_and_duplicates(df):
    """Check and handle missing values and duplicates."""
    print("[INFO] Missing values per column:\n", df.isnull().sum())
    print("[INFO] Duplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates()
    df = df.fillna(method='ffill')
    return df

def normalize_data(df, columns):
    """Normalize selected columns using min-max scaling."""
    data = df[columns].values
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    norm_data = (data - min_val) / (max_val - min_val)
    norm_df = pd.DataFrame(norm_data, columns=columns, index=df.index)
    return norm_df, min_val, max_val

def split_data(df, train_ratio=0.8):
    """Split data into train and validation sets."""
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    valid_df = df.iloc[train_size:]
    return train_df, valid_df

def save_preprocessed(train_df, valid_df, out_dir):
    """Save preprocessed train and validation data to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "X_train_preprocessed.csv")
    valid_path = os.path.join(out_dir, "X_valid_preprocessed.csv")
    train_df.to_csv(train_path)
    valid_df.to_csv(valid_path)
    print(f"[INFO] Saved train data to {train_path}")
    print(f"[INFO] Saved valid data to {valid_path}")

def automate_preprocessing(
    csv_path,
    out_dir,
    selected_columns=[
        "Bitcoin (USD)", "Ethereum (USD)", "Gold (USD per oz)",
        "Cardano (ADA)", "Binance Coin (BNB)", "Ripple (XRP)",
        "Dogecoin (DOGE)", "Solana (SOL)"
    ]
):
    print("[INFO] Loading dataset...")
    df = load_dataset(csv_path)
    print("[INFO] Checking missing values and duplicates...")
    df = check_missing_and_duplicates(df)
    print("[INFO] Normalizing selected columns...")
    norm_df, min_val, max_val = normalize_data(df, selected_columns)
    print("[INFO] Splitting data into train and validation sets...")
    train_df, valid_df = split_data(norm_df)
    print("[INFO] Saving preprocessed data...")
    save_preprocessed(train_df, valid_df, out_dir)
    print("[INFO] Preprocessing complete.")
    return train_df, valid_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True, help="Path to raw CSV dataset")
    parser.add_argument("--out-dir", required=True, help="Directory to save preprocessed data")
    args = parser.parse_args()

    automate_preprocessing(args.csv_path, args.out_dir)
