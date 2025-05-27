"""
Configuration settings for the financial transaction prediction model.
"""

# Data processing settings
DATA_CONFIG = {
    'date_column': 'Date',
    'amount_column': 'Amount',
    'category_column': 'Category',
    'transaction_type_column': 'Transaction Type',
    'description_column': 'Description',
    'train_test_split': 0.8,
    'random_state': 42
}

# Model training settings
MODEL_CONFIG = {
    'target_column': 'Amount',  # Column to predict
    'feature_columns': [
        'Category',
        'Transaction Type',
        'Description'
    ],
    'model_type': 'regression',  # or 'classification' for categorical predictions
    'validation_split': 0.2,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001
}

# Prediction settings
PREDICTION_CONFIG = {
    'forecast_horizon': 30,  # Number of days to forecast
    'confidence_interval': 0.95
}

# File paths
PATH_CONFIG = {
    'model_weights': 'model/weights/model_weights.h5',
    'scaler_path': 'model/weights/scaler.pkl',
    'encoder_path': 'model/weights/encoder.pkl',
    'model_config_path': 'model/weights/model_config.json'
} 