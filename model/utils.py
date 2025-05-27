"""
Utility functions for the financial transaction prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from datetime import datetime, timedelta
from model.model_config import DATA_CONFIG, PATH_CONFIG

def load_data(file_path):
    """
    Load and preprocess the transaction data.
    """
    df = pd.read_csv(file_path)
    df[DATA_CONFIG['date_column']] = pd.to_datetime(df[DATA_CONFIG['date_column']])
    return df

def prepare_features(df):
    """
    Prepare features for model training.
    """
    # Extract temporal features
    df['year'] = df[DATA_CONFIG['date_column']].dt.year
    df['month'] = df[DATA_CONFIG['date_column']].dt.month
    df['day'] = df[DATA_CONFIG['date_column']].dt.day
    df['day_of_week'] = df[DATA_CONFIG['date_column']].dt.dayofweek
    
    # Encode categorical features
    categorical_columns = ['Category', 'Transaction Type']
    encoders = {}
    
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    
    # Save encoders
    joblib.dump(encoders, PATH_CONFIG['encoder_path'])
    
    return df, encoders

def scale_features(df, feature_columns, scaler=None):
    """
    Scale numerical features.
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[feature_columns])
        joblib.dump(scaler, PATH_CONFIG['scaler_path'])
    else:
        scaled_data = scaler.transform(df[feature_columns])
    
    return scaled_data, scaler

def create_sequences(data, sequence_length):
    """
    Create sequences for time series prediction.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate model predictions.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

def save_model_config(config):
    """
    Save model configuration to file.
    """
    with open(PATH_CONFIG['model_config_path'], 'w') as f:
        json.dump(config, f, indent=4)

def load_model_config():
    """
    Load model configuration from file.
    """
    with open(PATH_CONFIG['model_config_path'], 'r') as f:
        return json.load(f) 