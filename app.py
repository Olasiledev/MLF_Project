#app.py
import os
from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("models/stock_lstm_model.h5")
scaler = joblib.load("models/scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict_stock_price():
  
    try:
        # Extracting input sequence
        input_sequence = request.json.get("sequence")
        if not input_sequence or len(input_sequence) != 60:
            return jsonify({"error": "Invalid input. Provide a sequence of 60 scaled values."}), 400

        input_array = np.array(input_sequence).reshape(1, 60, 1)

        #prediction
        prediction = model.predict(input_array)
        predicted_price = float(scaler.inverse_transform(prediction)[0][0])

        return jsonify({"predicted_price": predicted_price})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
