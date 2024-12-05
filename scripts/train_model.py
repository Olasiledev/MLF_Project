#scripts/train_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scripts.data_preprocessing import load_and_preprocess_data, create_sequences
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def create_sequences(data, seq_length):
    """
    Creating input sequences and labels for LSTM training.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_lstm_model(data_path, model_save_path, scaler_save_path):
    stock_data, scaler = load_and_preprocess_data(data_path)
    close_scaled = stock_data["Close_Scaled"].values

    # Creating sequences
    seq_length = 60
    X, y = create_sequences(close_scaled, seq_length)

    # Reshaping X for LSTM input (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Spliting into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    #LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)

    print(f"Model saved to -> {model_save_path}")
    print(f"Scaler saved to ->{scaler_save_path}")





def train_linear_regression(data):
    """
    Training Linear Regression model & compareing it with LSTM.
    """
     # Using indices as features
    X = np.arange(len(data["Close"])).reshape(-1, 1) 
    y = data["Close"].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Training Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Predicting and evaluating
    y_pred_lr = linear_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    print(f"Linear Regression RMSE-> {rmse_lr}")
    return linear_model, rmse_lr


if __name__ == "__main__":
    data, _ = load_and_preprocess_data("data/AAPL.csv")
    print("Training Linear Regression...")
    _, rmse_lr = train_linear_regression(data)
    print(f"Linear Regression RMSE: {rmse_lr}")

    print("Training LSTM...")
    train_lstm_model("data/AAPL.csv", "models/stock_lstm_model.h5", "models/scaler.pkl")
