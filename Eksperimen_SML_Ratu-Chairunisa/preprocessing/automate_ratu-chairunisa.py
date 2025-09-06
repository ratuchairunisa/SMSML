from pathlib import Path
import argparse
import logging
import json
from typing import Tuple, List
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import mlflow

DEFAULT_COLUMNS = [
    "Bitcoin (USD)", "Ethereum (USD)", "Cardano (ADA)",
    "Binance Coin (BNB)", "Ripple (XRP)", "Dogecoin (DOGE)",
    "Solana (SOL)", "Gold (USD per oz)"
]
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def load_csv(path: Path, date_col: str = "Date") -> pd.DataFrame:
    """Load CSV and return sorted DataFrame with Date index."""
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    return df

def ensure_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to full daily range and forward-fill missing values."""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx)
    df = df.ffill().bfill()  # fallback
    df.index.name = "Date"
    return df

def select_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Pick columns; raise if missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    return df[columns].copy()

def train_val_test_split_time_series(arr: np.ndarray, val_ratio=0.1, test_ratio=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous split for time-series (no shuffling)."""
    n = len(arr)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    train_end = n - (val_n + test_n)
    train = arr[:train_end]
    val = arr[train_end: train_end + val_n]
    test = arr[train_end + val_n:]
    return train, val, test

def fit_scaler(train_arr: np.ndarray) -> MinMaxScaler:
    """Fit MinMaxScaler on training set (2D flattening)."""
    # Flatten across time dimension: shape (-1, n_features)
    n_features = train_arr.shape[1]
    flat = train_arr.reshape(-1, n_features)
    scaler = MinMaxScaler()
    scaler.fit(flat)
    return scaler

def apply_scaler(arr: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Apply scaler to an array of shape (T, n_features)."""
    n_features = arr.shape[1]
    flat = arr.reshape(-1, n_features)
    flat_t = scaler.transform(flat)
    return flat_t.reshape(arr.shape)

def create_sequences(data: np.ndarray, n_past: int, n_future: int, step: int = 1):
    """
    Create sliding windows for multivariate forecasting.
    data: array shape (T, n_features)
    returns X: (n_samples, n_past, n_features), y: (n_samples, n_future, n_features)
    """
    X, y = [], []
    T = data.shape[0]
    for start in range(0, T - (n_past + n_future) + 1, step):
        X.append(data[start: start + n_past])
        y.append(data[start + n_past: start + n_past + n_future])
    return np.array(X), np.array(y)

def save_processed(dest: Path, **arrays):
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / "processed_data.npz"
    np.savez_compressed(out_file, **arrays)
    logging.info(f"Saved processed arrays to {out_file}")

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV")
    parser.add_argument("--output", type=str, default="processed", help="Output folder")
    parser.add_argument("--columns", type=str, default=",".join(DEFAULT_COLUMNS), help="Comma-separated columns to use")
    parser.add_argument("--n_past", type=int, default=30)
    parser.add_argument("--n_future", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)

    logging.info("Loading dataset...")
    df = load_csv(input_path)
    logging.info(f"Initial shape: {df.shape}")

    logging.info("Ensuring daily index and filling missing dates...")
    df = ensure_daily_index(df)

    columns = [c.strip() for c in args.columns.split(",")]
    logging.info(f"Selecting columns: {columns}")
    df = select_features(df, columns)

    arr = df.values  # shape (T, n_features)
    logging.info(f"Data array shape (T x F): {arr.shape}")

    # Train/Val/Test split on raw values (time-ordered)
    train_raw, val_raw, test_raw = train_val_test_split_time_series(arr, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    logging.info(f"Train/Val/Test lengths: {len(train_raw)}, {len(val_raw)}, {len(test_raw)}")

    # Fit scaler on train and transform
    scaler = fit_scaler(train_raw)
    train_scaled = apply_scaler(train_raw, scaler)
    val_scaled = apply_scaler(val_raw, scaler)
    test_scaled = apply_scaler(test_raw, scaler)

    # Save scaler
    scaler_path = out_dir / "scaler.save"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved scaler to: {scaler_path}")

    combined_scaled = np.vstack([train_scaled, val_scaled, test_scaled])
    logging.info(f"Combined scaled shape: {combined_scaled.shape}")

    # Create sequences
    X, Y = create_sequences(combined_scaled, n_past=args.n_past, n_future=args.n_future, step=1)
    logging.info(f"Created sequences: X {X.shape}, Y {Y.shape}")

    # Derive sample indices corresponding to splits
    n_train_time = train_scaled.shape[0]
    n_val_time = val_scaled.shape[0]
    X_train, Y_train = create_sequences(train_scaled, args.n_past, args.n_future, step=1)
    X_val, Y_val = create_sequences(val_scaled, args.n_past, args.n_future, step=1)
    X_test, Y_test = create_sequences(test_scaled, args.n_past, args.n_future, step=1)
    logging.info(f"Final splits (samples): train {X_train.shape[0]}, val {X_val.shape[0]}, test {X_test.shape[0]}")

    # Save processed arrays
    save_processed(out_dir, 
                   X_train=X_train, Y_train=Y_train,
                   X_val=X_val, Y_val=Y_val,
                   X_test=X_test, Y_test=Y_test)

    # Save metadata
    meta = {
        "columns": columns,
        "n_past": args.n_past,
        "n_future": args.n_future,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "input_rows": int(df.shape[0])
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logging.info("Saved metadata.json")

    # Save preprocessing report
    report = {
        "input_file": str(input_path),
        "columns": columns,
        "total_rows": int(df.shape[0]),
        "splits": {
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
        },
        "shapes": {
            "X_train": list(X_train.shape),
            "Y_train": list(Y_train.shape),
            "X_val": list(X_val.shape),
            "Y_val": list(Y_val.shape),
            "X_test": list(X_test.shape),
            "Y_test": list(Y_test.shape),
        }
    }
    with open(out_dir / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Saved preprocessing_report.json")

if __name__ == "__main__":
    main()

