#scripts/train_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scripts.data_preprocessing import load_and_preprocess_data, create_sequences
import joblib


def train_lstm_model(data_path, model_save_path, scaler_save_path):
   
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    # LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    model.save(model_save_path)
    print(f"LSTM model saved to: {model_save_path}")


def train_linear_regression(data_path):
   
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    # Prepare data for Linear Regression
    X = np.arange(len(y_train) + len(y_test)).reshape(-1, 1)
    y = np.concatenate([y_train, y_test])
    split_idx = len(y_train)

    X_train_lr, X_test_lr = X[:split_idx], X[split_idx:]
    y_train_lr, y_test_lr = y[:split_idx], y[split_idx:]

    # Training LR
    linear_model = LinearRegression()
    linear_model.fit(X_train_lr, y_train_lr)

    # Predictions and evaluation
    y_pred_lr = linear_model.predict(X_test_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test_lr, y_pred_lr))
    print(f"Linear Regression RMSE: {rmse_lr}")
    return linear_model, rmse_lr


if __name__ == "__main__":
    data_path = "data/AAPL.csv"
    lstm_model_path = "models/stock_lstm_model.h5"

    print("Training Linear Regression...")
    _, rmse_lr = train_linear_regression(data_path)
    print(f"Linear Regression RMSE: {rmse_lr}")

    print("Training LSTM...")
    train_lstm_model(data_path, lstm_model_path, None)
