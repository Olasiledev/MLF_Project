from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import numpy as np
from scripts.data_preprocessing import load_and_preprocess_data

file_path = "data/AAPL.csv"
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

def evaluate_models(lstm_model, lr_model, scaler, input_array, y_true):
    """
    Evaluate the LSTM and Linear Regression models on the given input array and return their RMSE values.

    Parameters:
    - lstm_model: Trained LSTM model
    - lr_model: Trained Linear Regression model
    - scaler: Scaler used to preprocess data (if applicable)
    - input_array: Input data for evaluation
    - y_true: True target value for comparison

    Returns:
    - dict with RMSE for both models
    """
    # Reshape input array for LSTM (assuming it's 1D with 60 features)
    input_array_lstm = input_array.reshape(1, 60, 1)

    # Evaluate LSTM model
    y_pred_lstm = lstm_model.predict(input_array_lstm)
    rmse_lstm = np.sqrt(mean_squared_error([y_true], y_pred_lstm))

    # Reshape input array for Linear Regression (1D with 60 features)
    input_array_lr = input_array.reshape(1, -1)

    # Scale the input if the scaler is used for Linear Regression
    if scaler:
        input_array_lr = scaler.transform(input_array_lr)

    # Evaluate Linear Regression model
    y_pred_lr = lr_model.predict(input_array_lr)
    rmse_lr = np.sqrt(mean_squared_error([y_true], y_pred_lr))

    return {"LSTM RMSE": rmse_lstm, "Linear Regression RMSE": rmse_lr}


#LSTM Model evaluation
lstm_model_path = 'models/optimized_lstm_model.h5'
lstm_model = load_model(lstm_model_path)
y_pred_lstm = lstm_model.predict(X_test)

lstm_mae = mean_absolute_error(y_test, y_pred_lstm)
lstm_mse = mean_squared_error(y_test, y_pred_lstm)
lstm_rmse = np.sqrt(lstm_mse)
lstm_r2 = r2_score(y_test, y_pred_lstm)

#Linear Regression model
lr_model = LinearRegression()
# Reshaping data
X_train_lr = X_train.reshape((X_train.shape[0], -1))
X_test_lr = X_test.reshape((X_test.shape[0], -1))

lr_model.fit(X_train_lr, y_train)
y_pred_lr = lr_model.predict(X_test_lr)

lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, y_pred_lr)



#comparison
print(f"LSTM Model Evaluation Metrics->")
print(f"MAE: {lstm_mae:.4f}")
print(f"MSE: {lstm_mse:.4f}")
print(f"RMSE: {lstm_rmse:.4f}")
print(f"R²: {lstm_r2:.4f}")
print("\n")

print(f"Linear Regression Model Evaluation Metrics->")
print(f"MAE: {lr_mae:.4f}")
print(f"MSE: {lr_mse:.4f}")
print(f"RMSE: {lr_rmse:.4f}")
print(f"R²: {lr_r2:.4f}")
