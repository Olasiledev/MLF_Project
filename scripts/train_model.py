#scripts/train_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scripts.data_preprocessing import load_and_preprocess_data, create_sequences
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def train_lstm_model(data_path, model_save_path, scaler_save_path=None):
 
    result = load_and_preprocess_data(data_path)

    if isinstance(result, tuple):
        stock_data = result[0]
        scaler = result[1] if len(result) > 1 else None
    else:
        stock_data = result
        scaler = None

    if isinstance(stock_data, pd.DataFrame):
        if "Close_Scaled" in stock_data.columns:
            close_scaled = stock_data["Close_Scaled"].values
        else:
            raise ValueError("'Close_Scaled' column not found.")
    elif isinstance(stock_data, np.ndarray):
        close_scaled = stock_data[:, -1]
    else:
        raise ValueError(f"Unexpected data type: {type(stock_data)}")

    seq_length = 60
    X, y = create_sequences(close_scaled, seq_length)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

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

    if scaler is not None and scaler_save_path:
        if isinstance(scaler, MinMaxScaler):
            joblib.dump(scaler, scaler_save_path)
            print(f"Scaler saved to -> {scaler_save_path}")
        else:
            print(f"Invalid scaler detected during training: {scaler}")

    print(f"LSTM model saved to -> {model_save_path}")



def train_linear_regression(data_path):
    result = load_and_preprocess_data(data_path)

    print("Stock data type->", type(result))

    if isinstance(result, tuple) and len(result) > 0:
        stock_data = result[0]
    else:
        stock_data = result

    if isinstance(stock_data, pd.DataFrame):
        if "Close" in stock_data.columns:
            y = stock_data["Close"].values
        else:
            raise ValueError("'Close' column not found.")
    elif isinstance(stock_data, np.ndarray):
        if stock_data.shape[1] > 0:
            print("Assuming last column 'Close'")
            y = stock_data[:, -1]
        else:
            raise ValueError("NumPy array is empty.")
    else:
        raise ValueError(f"Unexpected data type-> {type(stock_data)}")

    X = np.arange(len(y)).reshape(-1, 1)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    y_pred_lr = linear_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    print(f"Linear Regression RMSE: {rmse_lr}")
    return linear_model, rmse_lr


if __name__ == "__main__":
    data_path = "data/AAPL.csv"
    lstm_model_path = "models/stock_lstm_model.h5"
    scaler_save_path = "models/scaler.pkl"

    print("Training Linear Regression...")
    _, rmse_lr = train_linear_regression(data_path)
    print(f"Linear Regression RMSE: {rmse_lr}")

    print("Training LSTM...")
    train_lstm_model(data_path, lstm_model_path, scaler_save_path)
