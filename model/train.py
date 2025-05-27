"""
Script for training the financial transaction prediction model.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib

from model.model_config import DATA_CONFIG, MODEL_CONFIG, PATH_CONFIG
from model.utils import (
    load_data,
    prepare_features,
    scale_features,
    create_sequences,
    evaluate_predictions,
    save_model_config
)

def create_model(input_shape):
    """
    Create the LSTM model architecture.
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(data_path):
    """
    Train the model on the provided data.
    """
    # Create weights directory if it doesn't exist
    os.makedirs(os.path.dirname(PATH_CONFIG['model_weights']), exist_ok=True)
    
    # Load and prepare data
    df = load_data(data_path)
    df, encoders = prepare_features(df)
    
    # Prepare features and target
    feature_columns = MODEL_CONFIG['feature_columns'] + ['year', 'month', 'day', 'day_of_week']
    X = df[feature_columns].values
    y = df[MODEL_CONFIG['target_column']].values
    
    # Scale features
    X_scaled, scaler = scale_features(pd.DataFrame(X, columns=feature_columns), feature_columns)
    
    # Create sequences
    sequence_length = 7  # Use 7 days of history
    X_seq, y_seq = create_sequences(X_scaled, sequence_length)
    y_seq = y[sequence_length:]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq,
        test_size=1-DATA_CONFIG['train_test_split'],
        random_state=DATA_CONFIG['random_state']
    )
    
    # Create and train model
    model = create_model((sequence_length, X_seq.shape[2]))
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            PATH_CONFIG['model_weights'],
            save_best_only=True,
            monitor='val_loss'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=MODEL_CONFIG['epochs'],
        batch_size=MODEL_CONFIG['batch_size'],
        validation_split=MODEL_CONFIG['validation_split'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    
    # Save model configuration
    model_config = {
        'feature_columns': feature_columns,
        'sequence_length': sequence_length,
        'metrics': metrics,
        'training_history': {
            'loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1]
        }
    }
    save_model_config(model_config)
    
    return model, metrics

if __name__ == '__main__':
    # Example usage
    data_path = 'data/transactions_data.csv'  # Update with your data path
    model, metrics = train_model(data_path)
    print("Training completed. Model metrics:", metrics) 