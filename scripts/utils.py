#scripts/utils.py
import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date, save_path):
    """Fetching stock data from Yahoo Finance and saving as CSV file."""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.to_csv(save_path)
        print(f"Data saved to {save_path}")
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data -> {e}")
        raise
