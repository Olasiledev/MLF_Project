#scripts/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.data_preprocessing import load_and_preprocess_data

data, _ = load_and_preprocess_data("data/AAPL.csv")
print("Summary Statistics:")
print(data.describe())

#Closing Price
plt.figure(figsize=(10, 6))
data["Close"].plot(title="Closing Prices Over Time", xlabel="Date", ylabel="Price", grid=True)
plt.show()

#Histogram of Closing Prices
plt.figure(figsize=(10, 6))
data["Close"].hist(bins=50, color="skyblue", alpha=0.7)
plt.title("Distribution of Closing Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid()
plt.show()

#Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
