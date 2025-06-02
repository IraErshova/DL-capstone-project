import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from scipy import stats
import warnings

from model.preprocessing_data import main_preprocessing_pipeline, save_preprocessing_artifacts, load_preprocessing_artifacts

warnings.filterwarnings('ignore')

class FinancialForecastingModels:
    def __init__(self):
        self.expense_model = None
        self.income_model = None
        self.cashflow_model = None
        self.scaler = StandardScaler()
        self.mcc_encoder = LabelEncoder()
        self.evaluation_results = {}
        self.best_params = {}

    def create_sequences(self, df, client_id, sequence_length=30, forecast_horizon=7):
        """
        Create sequences for time series prediction (expenses only)
        """
        user_data = df[df['client_id'] == client_id].sort_values('date')

        if len(user_data) < sequence_length + forecast_horizon:
            return None, None

        # Features used in the input sequence
        features = [
            'amount', 'mcc_encoded', 'hour', 'day_of_week',
            'month', 'is_weekend'
        ]

        X, y = [], []

        for i in range(len(user_data) - sequence_length - forecast_horizon + 1):
            # Input sequence
            seq_data = user_data.iloc[i:i + sequence_length][features].values
            X.append(seq_data)

            # Forecast horizon target: future expense total
            future_data = user_data.iloc[i + sequence_length:i + sequence_length + forecast_horizon]
            future_expenses = future_data['amount'].sum()

            y.append([future_expenses])

        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape, output_dim=1):
        """
        Build LSTM model for expense forecasting
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(output_dim, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def build_gru_model(self, input_shape, output_dim=2):
        """
        Build GRU model (faster alternative)
        """
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(output_dim, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train_models(self):
        """
        Train all models using the new preprocessing pipeline
        """
        print("🔄 Loading preprocessed data...")

        # Get preprocessed data from the new pipeline
        data_dict = main_preprocessing_pipeline()

        # Save preprocessing artifacts for future use
        save_preprocessing_artifacts(data_dict)

        # Extract data from the dictionary
        X_train = data_dict['X_train']
        X_val = data_dict['X_val']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_val = data_dict['y_val']
        y_test = data_dict['y_test']
        user_train = data_dict['user_train']

        print(f"📊 Dataset shapes:")
        print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"   X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"   User features: {user_train.shape}")

        # Define callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train LSTM model
        print("\n🔧 Training LSTM model...")
        lstm_model = self.build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_dim=y_train.shape[1] if len(y_train.shape) > 1 else 1
        )

        history_lstm = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        self.save_models(lstm_model, 'lstm_model')

        # Evaluate LSTM model
        print("\n📊 Evaluating LSTM model...")
        lstm_results = self.evaluate_model_performance(
            lstm_model, X_test, y_test, "LSTM"
        )

        # Train GRU model for comparison
        print("\n🔧 Training GRU model...")
        gru_model = self.build_gru_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_dim=y_train.shape[1] if len(y_train.shape) > 1 else 1
        )

        history_gru = gru_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        self.save_models(gru_model, 'gru_model')

        # Evaluate GRU model
        print("\n📊 Evaluating GRU model...")
        gru_results = self.evaluate_model_performance(
            gru_model, X_test, y_test, "GRU"
        )


        print("\n⚠️  No user features available - skipping advanced model")
        histories = {
            'lstm': history_lstm,
            'gru': history_gru
        }

        comparison_results = self.compare_models(X_test, y_test)

        return {
            'histories': histories,
            'lstm_results': lstm_results,
            'gru_results': gru_results,
            'data_shapes': {
                'X_train': X_train.shape,
                'X_test': X_test.shape,
                'y_train': y_train.shape,
                'y_test': y_test.shape
            }
        }

    def predict_user_forecast(self, user_data_sequence, user_features=None):
        """
        Generate predictions for a user
        """
        # Load the model
        model = self.load_prediction_model()
        
        # Load preprocessing artifacts
        data_dict = load_preprocessing_artifacts()
        sequence_scaler = data_dict['scalers_encoders']['sequence_scaler']
        user_scaler = data_dict['scalers_encoders']['user_scaler']
        
        # Scale the input sequence
        user_data_scaled = sequence_scaler.transform(
            user_data_sequence.reshape(-1, user_data_sequence.shape[-1])
        ).reshape(user_data_sequence.shape)

        # Scale user features if available
        if user_features is not None:
            user_features_scaled = user_scaler.transform(user_features)
        else:
            user_features_scaled = None

        # Make prediction
        prediction = model.predict(user_data_scaled, verbose=0)

        # Convert NumPy types to Python native types
        return {
            'next_week_expenses': float(prediction[0][0])
        }

    def evaluate_model_performance(self, model, X_test, y_test, model_name="Model"):
        """
        Comprehensive model evaluation with multiple metrics
        """
        print(f"\n{'=' * 50}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'=' * 50}")

        # Make predictions
        y_pred = model.predict(X_test, verbose=0)

        # Handle multi-output models
        if len(y_pred.shape) > 2 or isinstance(y_pred, list):
            y_pred_expenses = y_pred[:, 0]
            y_test_expenses = y_test[:, 0]
        else:
            # Single output model
            y_pred_expenses = y_pred[:, 0] if y_pred.shape[1] > 1 else y_pred.flatten()
            y_test_expenses = y_test[:, 0] if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test.flatten()

        # Calculate metrics for expenses
        expense_metrics = self._calculate_metrics(y_test_expenses, y_pred_expenses, "Expenses")

        # Overall model performance
        overall_metrics = {
            'model_name': model_name,
            'expense_metrics': expense_metrics,
            'prediction_accuracy_grade': self._grade_model_performance(expense_metrics)
        }

        # Store results
        self.evaluation_results[model_name] = overall_metrics

        return overall_metrics

    def _calculate_metrics(self, y_true, y_pred, metric_type):
        """
        Calculate comprehensive metrics for predictions
        """
        # Remove any NaN or infinite values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            return {'error': 'No valid predictions'}

        # Core metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)

        # R-squared
        r2 = r2_score(y_true_clean, y_pred_clean)

        # MAPE (handling division by zero)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(np.abs(y_true_clean), 1e-8))) * 100

        # Direction accuracy (for financial predictions)
        direction_accuracy = np.mean(np.sign(y_true_clean[1:] - y_true_clean[:-1]) ==
                                     np.sign(y_pred_clean[1:] - y_pred_clean[:-1])) * 100

        # Prediction intervals
        residuals = y_true_clean - y_pred_clean
        prediction_std = np.std(residuals)

        # Statistical tests
        _, normality_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2_Score': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy,
            'Prediction_Std': prediction_std,
            'Residuals_Normal': normality_p > 0.05,
            'Mean_Actual': np.mean(y_true_clean),
            'Mean_Predicted': np.mean(y_pred_clean),
            'Correlation': np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
        }

        # Print metrics
        print(f"\n{metric_type} Prediction Metrics:")
        print(f"  MAE (Mean Absolute Error): ${mae:.2f}")
        print(f"  RMSE (Root Mean Square Error): ${rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print(f"  Direction Accuracy: {direction_accuracy:.2f}%")
        print(f"  Prediction Standard Deviation: ${prediction_std:.2f}")
        print(f"  Actual vs Predicted Correlation: {metrics['Correlation']:.4f}")

        return metrics

    def _grade_model_performance(self, expense_metrics):
        """
        Grade model performance from A+ to F
        """
        if 'error' in expense_metrics:
            return 'F'

        # Scoring criteria
        score = 0

        # R² Score (40% weight)
        r2_expense = expense_metrics.get('R2_Score', 0)
        if r2_expense > 0.9:
            score += 40
        elif r2_expense > 0.8:
            score += 35
        elif r2_expense > 0.7:
            score += 30
        elif r2_expense > 0.6:
            score += 25
        elif r2_expense > 0.5:
            score += 20
        elif r2_expense > 0.3:
            score += 15
        elif r2_expense > 0.1:
            score += 10
        else:
            score += 5

        # MAPE (30% weight)
        mape_expense = expense_metrics.get('MAPE', 100)
        if mape_expense < 5:
            score += 30
        elif mape_expense < 10:
            score += 25
        elif mape_expense < 15:
            score += 20
        elif mape_expense < 20:
            score += 15
        elif mape_expense < 30:
            score += 10
        else:
            score += 5

        # Direction Accuracy (20% weight)
        dir_acc = expense_metrics.get('Direction_Accuracy', 0)
        if dir_acc > 80:
            score += 20
        elif dir_acc > 70:
            score += 15
        elif dir_acc > 60:
            score += 10
        elif dir_acc > 50:
            score += 5

        # Correlation (10% weight)
        corr = expense_metrics.get('Correlation', 0)
        if corr > 0.9:
            score += 10
        elif corr > 0.8:
            score += 8
        elif corr > 0.7:
            score += 6
        elif corr > 0.6:
            score += 4
        elif corr > 0.5:
            score += 2

        # Grade assignment
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        elif score >= 45:
            return 'D+'
        elif score >= 40:
            return 'D'
        else:
            return 'F'

    def cross_validate_model(self, X, y, model_type='LSTM', n_splits=5):
        """
        Time series cross-validation
        """
        print(f"\n{'=' * 50}")
        print(f"CROSS-VALIDATION FOR {model_type}")
        print(f"{'=' * 50}")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {'mae': [], 'mse': [], 'r2': []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"Fold {fold + 1}/{n_splits}")

            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            # Build model
            if model_type == 'LSTM':
                model = self.build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
            else:
                model = self.build_gru_model(input_shape=(X.shape[1], X.shape[2]))

            # Train
            model.fit(
                X_train_cv, y_train_cv,
                epochs=30,
                batch_size=32,
                verbose=0,
                callbacks=[EarlyStopping(patience=10)]
            )

            # Predict and evaluate
            y_pred_cv = model.predict(X_val_cv, verbose=0)

            # Calculate metrics for expenses (first output)
            mae = mean_absolute_error(y_val_cv[:, 0], y_pred_cv[:, 0])
            mse = mean_squared_error(y_val_cv[:, 0], y_pred_cv[:, 0])
            r2 = r2_score(y_val_cv[:, 0], y_pred_cv[:, 0])

            cv_scores['mae'].append(mae)
            cv_scores['mse'].append(mse)
            cv_scores['r2'].append(r2)

            print(f"  MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.4f}")

        # Summary
        print(f"\nCross-Validation Results:")
        print(f"  MAE: {np.mean(cv_scores['mae']):.2f} ± {np.std(cv_scores['mae']):.2f}")
        print(f"  MSE: {np.mean(cv_scores['mse']):.2f} ± {np.std(cv_scores['mse']):.2f}")
        print(f"  R²:  {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")

        return cv_scores

    def compare_models(self, X_test, y_test):
        """
        Compare all trained models
        """
        print(f"\n{'=' * 60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'=' * 60}")

        comparison_data = []

        for model_name, results in self.evaluation_results.items():
            expense_metrics = results['expense_metrics']
            grade = results['prediction_accuracy_grade']

            comparison_data.append({
                'Model': model_name,
                'Grade': grade,
                'R²': expense_metrics.get('R2_Score', 0),
                'RMSE': expense_metrics.get('RMSE', 0),
                'MAPE': expense_metrics.get('MAPE', 0),
                'Direction_Acc': expense_metrics.get('Direction_Accuracy', 0)
            })

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R²', ascending=False)

        print(comparison_df.to_string(index=False, float_format='%.3f'))

        # Recommendation
        best_model = comparison_df.iloc[0]['Model']
        print(f"\n🏆 RECOMMENDED MODEL: {best_model}")
        print(f"   Grade: {comparison_df.iloc[0]['Grade']}")
        print(f"   R² Score: {comparison_df.iloc[0]['R²']:.4f}")

        return comparison_df

    def save_models(self, model, model_name, model_dir='models/'):
        """
        Save trained models
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        model.save(f'{model_dir}/{model_name}.keras')

    def load_models(self, model_dir='models/'):
        """
        Load pre-trained models
        """
        import joblib

        self.expense_model = tf.keras.models.load_model(f'{model_dir}/lstm_expense_model.h5')
        self.cashflow_model = tf.keras.models.load_model(f'{model_dir}/advanced_cashflow_model.h5')
        self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
        self.mcc_encoder = joblib.load(f'{model_dir}/mcc_encoder.pkl')

    def load_prediction_model(self):
        return tf.keras.models.load_model('models/lstm_model.keras')


# Usage example with comprehensive evaluation
if __name__ == "__main__":
    # Initialize the model class
    forecaster = FinancialForecastingModels()

    # Example usage with evaluation
    print("🚀 Financial Forecasting Models with Comprehensive Evaluation")
    print("=" * 60)

    # Train models with evaluation
    print("🔧 Training models...")
    results = forecaster.train_models()
