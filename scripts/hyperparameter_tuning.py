#scripts/hyperparameter_tuning.py
import keras_tuner as kt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scripts.data_preprocessing import load_and_preprocess_data
import json
import os

file_path = "data/AAPL.csv"

X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

# Function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(
        LSTM(
            units=hp.Int("units", min_value=50, max_value=200, step=50),
            activation="relu",
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    model.add(Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
        loss="mean_squared_error",
    )
    return model

# Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=10,
    factor=3,
    directory='hyperparameter_tuning',
    project_name='lstm_tuning'
)

# Searching with validation data
tuner.search(X_train, y_train, validation_data=(X_test, y_test))


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Saving hyperparameters
config_path = 'configs/best_hyperparameters.json'
os.makedirs(os.path.dirname(config_path), exist_ok=True)
with open(config_path, 'w') as f:
    json.dump({
        'units': best_hps.get('units'),
        'dropout': best_hps.get('dropout'),
        'learning_rate': best_hps.get('learning_rate')
    }, f)
print(f"hyperparameters saved to {config_path}")

# Retraining with best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

model_path = 'models/optimized_lstm_model.h5'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
best_model.save(model_path)
print(f"Optimized model saved to {model_path}")
