import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
data = np.load("processed/processed_data.npz")
X_train, Y_train = data['X_train'], data['Y_train']
X_val, Y_val = data['X_val'], data['Y_val']
X_test, Y_test = data['X_test'], data['Y_test']  # tambahkan untuk evaluasi

# Ambil dimensi
n_past, n_features = X_train.shape[1], X_train.shape[2]
n_future = Y_train.shape[1]

# Bangun model sesuai manual.ipynb
def build_model(n_past, n_features, n_future):
    model = Sequential([
        LSTM(128, activation="tanh", return_sequences=True, input_shape=(n_past, n_features)),
        Dropout(0.2),
        LSTM(64, activation="tanh"),
        Dense(n_future * n_features)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"]
    )
    return model

# Callback
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Build & train
model = build_model(n_past, n_features, n_future)

history = model.fit(
    X_train, Y_train.reshape(len(Y_train), -1),
    validation_data=(X_val, Y_val.reshape(len(Y_val), -1)),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluasi
def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    y_true = Y_test.reshape(len(Y_test), -1)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    return mae, mse

print("\nEvaluasi Model:")
mae, mse = evaluate_model(model, X_test, Y_test)
