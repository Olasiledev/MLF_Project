# app.py
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from scripts.data_preprocessing import load_and_preprocess_data
from scripts.model_comparison import evaluate_models  # Import the function
import joblib
import os

app = Flask(__name__)

# Load models
lstm_model_path = 'models/optimized_lstm_model.h5'
lstm_model = load_model(lstm_model_path)

lr_model_path = 'models/linear_regression_model.pkl'
lr_model = joblib.load(lr_model_path)

# Load scaler
scaler_path = 'models/scaler.pkl'
scaler = joblib.load(scaler_path)

# Load and preprocess data
file_path = "data/AAPL.csv"
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

@app.route("/predict", methods=["POST"])
def predict_stock_price():
    try:
        # Extracting input sequence
        input_sequence = request.json.get("sequence")
        if not input_sequence or len(input_sequence) != 60:
            return jsonify({"error": "Invalid input. Provide a sequence of 60 scaled values."}), 400

        # Convert input sequence to a NumPy array and reshape for evaluation
        input_array = np.array(input_sequence)

        # Select a true value from the test data for comparison
        y_true = y_test[0]

        # Evaluate the models with the input array and true value
        results = evaluate_models(lstm_model, lr_model, scaler, input_array, y_true)

        # Determine the better model based on RMSE
        best_model_name = min(results, key=results.get)  # Model with the lowest RMSE

        # Make predictions using the best model
        if best_model_name == "LSTM RMSE":
            prediction = lstm_model.predict(input_array.reshape(1, 60, 1))
        else:
            prediction = lr_model.predict(input_array.reshape(1, -1))
            prediction = scaler.inverse_transform(prediction.reshape(1, -1))  # Inverse scale if needed

        predicted_price = float(prediction[0])

        return jsonify({
            "model_used": best_model_name.split()[0],
            "predicted_price": predicted_price,
            "evaluation_results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
