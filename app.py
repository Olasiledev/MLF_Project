import os
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from scripts.data_preprocessing import load_and_preprocess_data
from scripts.model_comparison import lr_rmse, lstm_rmse
import joblib

app = Flask(__name__)

# Load models
lstm_model_path = 'models/optimized_lstm_model.h5'
lstm_model = load_model(lstm_model_path)

lr_model_path = 'models/linear_regression_model.pkl'
lr_model = joblib.load(lr_model_path)

# scaler
scaler_path = 'models/scaler.pkl'
scaler = joblib.load(scaler_path)

# Loading and preprocessing data
file_path = "data/AAPL.csv"


# Select best model
if lstm_rmse < lr_rmse:
    best_model = lstm_model
    best_model_name = "LSTM"
else:
    best_model = lr_model
    best_model_name = "LinearRegression"


@app.route("/predict", methods=["POST"])
def predict_stock_price():
    try:
        # Extracting input sequence
        input_sequence = request.json.get("sequence")
        if not input_sequence or len(input_sequence) != 60:
            return jsonify({"error": "Invalid input. Provide a sequence of 60 scaled values."}), 400

        # Reshaping and scaling input for the LSTM model
        input_array = np.array(input_sequence).reshape(1, 60, 1)
        

        # Prediction
        if best_model_name == "LSTM":
            prediction = best_model.predict(input_array)
        elif best_model_name == "LinearRegression":
            # Reshaping input for Linear Regression
            input_array = np.arange(len(input_array)).reshape(-1, 1)

            input_array = scaler.transform(input_array)  

            prediction = best_model.predict(input_array)
        else:
            return jsonify({"error": "No suitable model selected."}), 500

        predicted_price = float(prediction[0])

        return jsonify({
            "model_used": best_model_name,
            "predicted_price": predicted_price
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host="0.0.0.0", port=port)