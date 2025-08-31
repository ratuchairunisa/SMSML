import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# 1. Download dataset (example using KaggleHub, adjust as needed)
def download_dataset():
    try:
        import kagglehub
        path = kagglehub.dataset_download("muhamedumarjamil/crypto-and-gold-prices-dataset-20152025")
        print("Dataset downloaded to:", path)
        return path
    except ImportError:
        print("kagglehub not installed. Please install it or download the dataset manually.")
        return None

# 2. Load dataset
def load_dataset(csv_path):
    df = pd.read_csv(r'D:\SMSML_Ratu_Chairunisa\crypto.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# 3. EDA: Plot trends
def plot_trends(df):
    plt.figure(figsize=(14, 7))
    symbols = ["Bitcoin (USD)", "Ethereum (USD)", "Gold (USD per oz)"]
    for symbol in symbols:
        plt.plot(df.index, df[symbol], label=symbol)
    plt.title("Tren Harga Close Bitcoin, Etherum, & Emas (2015–2025)")
    plt.xlabel("Tahun")
    plt.ylabel("Harga (USD)")
    plt.legend()
    plt.show()

# 4. Data Preprocessing
def preprocess_data(df):
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates()
    df = df.fillna(method='ffill')
    return df

def normalize_series(data, min_val, max_val):
    return (data - min_val) / max_val

# 5. Prepare data for modeling
def prepare_data(df, selected_columns):
    data = df[selected_columns].values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))
    train_size = int(len(data) * 0.8)
    x_train, x_valid = data[:train_size], data[train_size:]
    return x_train, x_valid, data

def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

# 6. Build and train model
def build_model(N_PAST, N_FEATURES, N_FUTURE):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(N_PAST, N_FEATURES)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(N_FUTURE * N_FEATURES),
        tf.keras.layers.Reshape((N_FUTURE, N_FEATURES))
    ])
    return model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('mae') < 0.055 and logs.get('val_mae') < 0.055:
            self.model.stop_training = True

def main():
    # Download and load dataset
    dataset_path = download_dataset()
    if dataset_path is None:
        return
    csv_path = os.path.join(dataset_path, "Crypto Data Since 2015.csv")
    df = load_dataset(csv_path)

    # EDA
    plot_trends(df)

    # Preprocessing
    df = preprocess_data(df)

    # Feature selection
    selected_columns = [
        "Bitcoin (USD)", "Ethereum (USD)", "Gold (USD per oz)",
        "Cardano (ADA)", "Binance Coin (BNB)", "Ripple (XRP)",
        "Dogecoin (DOGE)", "Solana (SOL)"
    ]
    N_FEATURES = len(selected_columns)
    x_train, x_valid, data = prepare_data(df, selected_columns)

    # Windowed dataset
    BATCH_SIZE = 32
    N_PAST = 30
    N_FUTURE = 30
    SHIFT = 1
    train_set = windowed_dataset(x_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)
    valid_set = windowed_dataset(x_valid, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)

    # Model
    model = build_model(N_PAST, N_FEATURES, N_FUTURE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mae', optimizer=optimizer, metrics=["mae"])
    callbacks = myCallback()
    model.fit(train_set, validation_data=valid_set, epochs=100, callbacks=[callbacks], verbose=1)

    # Prediction and visualization
    crypto_names = ["BTC", "ETH", "GOLD", "ADA", "BNB", "XRP", "DOGE", "SOL"]
    train_pred = model.predict(train_set)
    predictions = train_pred[0]
    last_date = pd.to_datetime("2025-07-24")
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions), freq="D")
    n_crypto = len(crypto_names)
    n_cols = 2
    n_rows = int(np.ceil(n_crypto / n_cols))
    plt.figure(figsize=(14, 3 * n_rows))
    for i, crypto in enumerate(crypto_names):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(future_dates, predictions[:, i], marker="o", linewidth=2, label=crypto, color="tab:red")
        plt.title(f"Prediksi {crypto} 30 Hari ke Depan (Mulai 25 Juli 2025)")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga (Normalized)")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()