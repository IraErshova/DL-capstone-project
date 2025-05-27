"""
Script for making predictions using the trained model.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime, timedelta

from model.model_config import DATA_CONFIG, MODEL_CONFIG, PATH_CONFIG, PREDICTION_CONFIG
from model.utils import (
    load_data,
    prepare_features,
    scale_features,
    create_sequences,
    load_model_config
)

class TransactionPredictor:
    def __init__(self):
        """
        Initialize the predictor with the trained model and necessary components.
        """
        self.model = tf.keras.models.load_model(PATH_CONFIG['model_weights'])
        self.config = load_model_config()
        self.scaler = joblib.load(PATH_CONFIG['scaler_path'])
        self.encoders = joblib.load(PATH_CONFIG['encoder_path'])
        
    def prepare_input_data(self, df):
        """
        Prepare input data for prediction.
        """
        # Prepare features
        df, _ = prepare_features(df)
        
        # Get feature columns
        feature_columns = self.config['feature_columns']
        X = df[feature_columns].values
        
        # Scale features
        X_scaled, _ = scale_features(pd.DataFrame(X, columns=feature_columns), feature_columns, self.scaler)
        
        return X_scaled
    
    def predict_next_day(self, recent_data):
        """
        Predict the next day's transaction amount.
        """
        # Prepare input data
        X = self.prepare_input_data(recent_data)
        
        # Create sequence
        sequence_length = self.config['sequence_length']
        if len(X) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} days of data for prediction")
        
        X_seq = X[-sequence_length:].reshape(1, sequence_length, X.shape[1])
        
        # Make prediction
        prediction = self.model.predict(X_seq)[0][0]
        
        return prediction
    
    def forecast(self, recent_data, days=PREDICTION_CONFIG['forecast_horizon']):
        """
        Forecast transaction amounts for the next n days.
        """
        predictions = []
        current_data = recent_data.copy()
        
        for _ in range(days):
            # Predict next day
            next_day_pred = self.predict_next_day(current_data)
            predictions.append(next_day_pred)
            
            # Add prediction to current data for next iteration
            next_date = current_data[DATA_CONFIG['date_column']].max() + timedelta(days=1)
            new_row = current_data.iloc[-1].copy()
            new_row[DATA_CONFIG['date_column']] = next_date
            new_row[MODEL_CONFIG['target_column']] = next_day_pred
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # Create forecast dataframe
        forecast_dates = pd.date_range(
            start=recent_data[DATA_CONFIG['date_column']].max() + timedelta(days=1),
            periods=days
        )
        
        forecast_df = pd.DataFrame({
            DATA_CONFIG['date_column']: forecast_dates,
            MODEL_CONFIG['target_column']: predictions
        })
        
        return forecast_df
    
    def get_confidence_intervals(self, forecast_df):
        """
        Calculate confidence intervals for the forecast.
        """
        confidence = PREDICTION_CONFIG['confidence_interval']
        std_dev = np.std(forecast_df[MODEL_CONFIG['target_column']])
        
        forecast_df['lower_bound'] = forecast_df[MODEL_CONFIG['target_column']] - (1.96 * std_dev)
        forecast_df['upper_bound'] = forecast_df[MODEL_CONFIG['target_column']] + (1.96 * std_dev)
        
        return forecast_df

def load_predictor():
    """
    Load the trained predictor.
    """
    return TransactionPredictor()

if __name__ == '__main__':
    # Example usage
    data_path = 'data/transactions_data.csv'  # Update with your data path
    recent_data = load_data(data_path)
    
    predictor = load_predictor()
    forecast = predictor.forecast(recent_data)
    forecast_with_intervals = predictor.get_confidence_intervals(forecast)
    
    print("Forecast for next", PREDICTION_CONFIG['forecast_horizon'], "days:")
    print(forecast_with_intervals) 