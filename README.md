# Stock Price Prediction Project

## Project Overview

This project uses machine learning techniques to predict stock prices using historical stock data.

## Project Structure
StockPricePrediction/
│
├── app.py                   
├── data/
│   └── AAPL.csv 
├── models/
│   ├── stock_lstm_model.h5
│   └── scaler.pkl
├── scripts/
│   ├── data_preprocessing.py 
│   ├── eda.py  
│   ├── hyperparameter_tuning.py
│   ├── model_evaluation.py
│   └── model_comparison.py 
├── configs/
│   └── best_hyperparameters.json
├── requirements.txt   
└── README.md      

### 1. Clone repository

cd StockPricePrediction

### 2. Set up VE 

### Install dependencies : pip install -r requirements.txt

### To run all scripts and start flask app run : ./start.sh 

### To run scripts individually ------> 

### run for data preprocessing : python3 -m scripts.data_preprocessing
 
### run for Exploratory Data Analysis (EDA) : python3 -m scripts.eda

### run for Hyperparameter Tuning: python3 -m scripts.hyperparameter_tuning

### run for Model Evaluation : python3 -m scripts.model_evaluation

### Model Comparison run: python3 -m scripts.model_comparison

### FLASK APP Deployment run: python3 app.py

### local testing
url: http://127.0.0.1:8000/predict

request body sample:
{
    "sequence": [0.003, 0.004, 0.005, 0.003, 0.006, 0.004, 0.005, 0.003, 0.006, 0.004, 
                 0.005, 0.003, 0.006, 0.004, 0.005, 0.003, 0.006, 0.004, 0.005, 0.003, 
                 0.006, 0.004, 0.005, 0.003, 0.006, 0.004, 0.005, 0.003, 0.006, 0.004, 
                 0.005, 0.003, 0.006, 0.004, 0.005, 0.003, 0.006, 0.004, 0.005, 0.003, 
                 0.006, 0.004, 0.005, 0.003, 0.006, 0.004, 0.005, 0.003, 0.006, 0.004, 
                 0.005, 0.003, 0.006, 0.004, 0.005, 0.003, 0.006, 0.004, 0.005, 0.003]
}

sample output: 
{
  "predicted_price": 4.7739177703857
}

