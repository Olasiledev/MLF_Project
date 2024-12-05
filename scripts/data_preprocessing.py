#scripts/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np


def create_sequences(data, seq_length):
  
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X).reshape(-1, seq_length, 1), np.array(y)



def load_and_preprocess_data(file_path, seq_length=50, test_size=0.2):
 
    try:
        stock_data = pd.read_csv(file_path, skiprows=3, header=None)
        stock_data.columns = ["Date", "Price", "Adj Close", "Close", "High", "Low", "Volume"]

        stock_data["Date"] = pd.to_datetime(stock_data["Date"], errors="coerce")
        stock_data.dropna(subset=["Date"], inplace=True)
        stock_data.set_index("Date", inplace=True)

        stock_data.ffill(inplace=True)

        #'Close' column scaling
        scaler = MinMaxScaler()
        stock_data["Close_Scaled"] = scaler.fit_transform(stock_data["Close"].astype(float).values.reshape(-1, 1))

        #sequences for LSTM
        X, y = create_sequences(stock_data["Close_Scaled"].values, seq_length)

        # Spliting into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        print("Data preprocessing completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error preprocessing: {e}")
        raise
