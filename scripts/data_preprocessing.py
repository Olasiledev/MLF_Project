#scripts/data_preprocessing.py
import os
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np


def fetch_and_save_stock_data(ticker, start_date, end_date, file_path):

    try:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        stock_data.to_csv(file_path)
        print(f"Data for {ticker} saved to {file_path}")
    except Exception as e:
        print(f"Error fetching data-> {e}")
        raise


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X).reshape(-1, seq_length, 1), np.array(y)


def load_and_preprocess_data(file_path, seq_length=50, test_size=0.2):

    try:
        if not os.path.exists(file_path):
            print(f"{file_path} not found. Fetching and saving data...")
            fetch_and_save_stock_data("AAPL", "2020-01-01", "2024-12-31", file_path)

        stock_data = pd.read_csv(file_path, skiprows=3, header=None)
        stock_data.columns = ["Date", "Price", "Adj Close", "Close", "High", "Low", "Volume"]

        stock_data["Date"] = pd.to_datetime(stock_data["Date"], errors="coerce")
        stock_data.dropna(subset=["Date"], inplace=True)
        stock_data.set_index("Date", inplace=True)

        stock_data.ffill(inplace=True)

        scaler = MinMaxScaler()
        stock_data["Close_Scaled"] = scaler.fit_transform(stock_data["Close"].astype(float).values.reshape(-1, 1))

        # Creating sequences for LSTM
        X, y = create_sequences(stock_data["Close_Scaled"].values, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        print("Data preprocessing completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error preprocessing: {e}")
        raise


if __name__ == "__main__":
    file_path = "data/AAPL.csv"

    print("Starting data preprocessing...")
    load_and_preprocess_data(file_path)
