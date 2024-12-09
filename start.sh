#!/bin/bash

echo "Running preprocessing scripts..."
python3 -m scripts.data_preprocessing
python3 -m scripts.hyperparameter_tuning
python3 -m scripts.model_evaluation
python3 -m scripts.model_comparison
python3 -m scripts.train_model


echo "Starting the application with Gunicorn..."
gunicorn app:app
