#scripts/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.data_preprocessing import fetch_and_save_stock_data

file_path = "data/AAPL.csv"

try:
    print("Loading data...")
    data = pd.read_csv(file_path, skiprows=1)

    required_columns = ["Date", "Price", "Adj Close", "Close", "High", "Low", "Volume"]
    current_columns = data.columns.tolist()

    if len(current_columns) < len(required_columns):
        raise ValueError(
            f"CSV has fewer columns ({len(current_columns)}) than required ({len(required_columns)})."
        )

    data.columns = required_columns[: len(current_columns)]

    # Checking if 'Date' column exists and process it
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data.dropna(subset=["Date"], inplace=True) 
        data.set_index("Date", inplace=True) 
    else:
        raise ValueError("'Date' column not found in the dataset.")

    print("Data successfully loaded and parsed.")

except FileNotFoundError:
    print("File not found. Fetching data...")
    fetch_and_save_stock_data("AAPL", "2020-01-01", "2024-12-31", file_path)
    data = pd.read_csv(file_path, skiprows=1)
    data.columns = required_columns[: len(data.columns)]
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data.dropna(subset=["Date"], inplace=True)
    data.set_index("Date", inplace=True)

except Exception as e:
    print(f"Unexpected error: {e}")
    exit()

print(data.head())

# Summary Statistics
print("Summary Statistics:")
print(data.describe())

plt.figure(figsize=(10, 6))
if "Close" in data.columns:
    data["Close"].plot(title="Closing Prices Over Time", xlabel="Date", ylabel="Price", grid=True)
    plt.show()
else:
    print("'Close' column not found. Skipping time series plot.")

# Histogram of Closing Prices
plt.figure(figsize=(10, 6))
if "Close" in data.columns:
    data["Close"].hist(bins=50, color="skyblue", alpha=0.7)
    plt.title("Distribution of Closing Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()
else:
    print("'Close' column not found. Skipping histogram.")

# Correlation Heatmap
plt.figure(figsize=(10, 6))
if data.select_dtypes(include=["float64", "int64"]).shape[1] > 1:  
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("Insufficient numeric data for correlation heatmap.")

print("EDA completed successfully.")
