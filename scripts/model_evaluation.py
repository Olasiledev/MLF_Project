#scripts/model_evaluation.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from scripts.data_preprocessing import load_and_preprocess_data

file_path = "data/AAPL.csv"
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

model_path = 'models/optimized_lstm_model.h5'
model = load_model(model_path)

y_pred = model.predict(X_test)

#Evaluating metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Metrics->")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
