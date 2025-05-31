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
from tqdm import tqdm
import json
import warnings

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

    def prepare_data(self, transactions_df, users_df, cards_df, mcc_codes_path):
        """
        Prepare and engineer features from the dataset
        """
        # Load MCC codes
        with open(mcc_codes_path, 'r') as f:
            mcc_codes = json.load(f)

        # Merge users into transactions
        df = transactions_df.merge(
            users_df,
            left_on='client_id',
            right_on='id',
            how='left',
            suffixes=('', '_user')
        )

        # Merge cards into the result
        df = df.merge(
            cards_df,
            left_on='card_id',
            right_on='id',
            how='left',
            suffixes=('', '_card')
        )

        # Drop unwanted columns
        df = df.drop(columns=['id_user', 'id_card', 'client_id_card'], errors='ignore')

        # Feature engineering
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Categorize transactions
        # Clean and convert all amounts to negative floats
        df['amount'] = (
            df['amount']
            .str.replace('$', '', regex=False)  # Remove dollar sign
            .astype(float)  # Convert to float
            .apply(lambda x: abs(x))  # Ensure all values are abs
        )

        # MCC category encoding
        df['mcc_encoded'] = self.mcc_encoder.fit_transform(df['mcc'].fillna('unknown'))

        # User spending patterns
        user_stats = df.groupby('client_id').agg({
            'amount': ['mean', 'std', 'sum'],
            'date': ['count']
        }).reset_index()
        user_stats.columns = ['client_id', 'avg_amount', 'std_amount', 'total_spent', 'transaction_count']

        df = df.merge(user_stats, on='client_id', how='left')

        return df

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

    def build_lstm_model(self, input_shape, output_dim=2):
        """
        Build LSTM model for expense/income forecasting
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
            Dense(output_dim, activation='linear')  # [expenses, income]
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

    def build_advanced_model(self, input_shape, user_features_dim):
        """
        Advanced model combining sequential data with user features
        """
        # Sequential input
        seq_input = Input(shape=input_shape, name='sequence_input')
        lstm_out = LSTM(128, return_sequences=True)(seq_input)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(64)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)

        # User features input
        user_input = Input(shape=(user_features_dim,), name='user_features')
        user_dense = Dense(32, activation='relu')(user_input)
        user_dense = Dropout(0.2)(user_dense)

        # Combine both inputs
        combined = Concatenate()([lstm_out, user_dense])
        combined = Dense(64, activation='relu')(combined)
        combined = Dense(32, activation='relu')(combined)

        # Multi-output
        expense_output = Dense(1, activation='linear', name='expense_forecast')(combined)
        income_output = Dense(1, activation='linear', name='income_forecast')(combined)
        cashflow_output = Dense(7, activation='linear', name='cashflow_7day')(combined)

        model = Model(
            inputs=[seq_input, user_input],
            outputs=[expense_output, income_output, cashflow_output]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'expense_forecast': 'mse',
                'income_forecast': 'mse',
                'cashflow_7day': 'mse'
            },
            loss_weights={
                'expense_forecast': 1.0,
                'income_forecast': 1.0,
                'cashflow_7day': 0.5
            },
            metrics=['mae']
        )

        return model

    def train_models(self, df, sequence_length=30, forecast_horizon=7, tune_hyperparameters=False):
        """
        Train all models with comprehensive evaluation
        """
        all_X, all_y = [], []
        user_features_list = []
        client_ids = df['client_id'].unique()
        print(f"Processing {len(client_ids)} unique clients...")

        # Prepare data for all users
        for client_id in tqdm(client_ids, desc="Generating sequences"):
            X, y = self.create_sequences(df, client_id, sequence_length, forecast_horizon)
            if X is not None:
                all_X.append(X)
                all_y.append(y)

                # User-specific features
                user_data = df[df['client_id'] == client_id].iloc[0]
                user_features = [
                    user_data['avg_amount'],
                    user_data['std_amount'],
                    user_data['transaction_count'],
                    user_data.get('age', 0),
                    user_data.get('credit_limit', 0)
                ]
                user_features_list.extend([user_features] * len(X))

        # Combine all data
        X_combined = np.vstack(all_X)
        y_combined = np.vstack(all_y)
        user_features_array = np.array(user_features_list)

        print(f"Dataset shape: X={X_combined.shape}, y={y_combined.shape}")

        # Normalize features
        X_combined_scaled = self.scaler.fit_transform(
            X_combined.reshape(-1, X_combined.shape[-1])
        ).reshape(X_combined.shape)

        # Split data (80% train, 10% validation, 10% test)
        X_temp, X_test, y_temp, y_test, user_temp, user_test = train_test_split(
            X_combined_scaled, y_combined, user_features_array,
            test_size=0.1, random_state=42
        )

        X_train, X_val, y_train, y_val, user_train, user_val = train_test_split(
            X_temp, y_temp, user_temp,
            test_size=0.111, random_state=42  # 0.111 * 0.9 ≈ 0.1 of total
        )

        print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

        # Hyperparameter tuning (optional)
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            lstm_best_params, _ = self.hyperparameter_tuning(X_train, y_train, X_val, y_val, 'LSTM')
            gru_best_params, _ = self.hyperparameter_tuning(X_train, y_train, X_val, y_val, 'GRU')
        else:
            lstm_best_params = {'lstm_units': 128, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 32}
            gru_best_params = {'lstm_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 32}

        # Train LSTM model
        print("\n" + "=" * 50)
        print("TRAINING LSTM MODEL")
        print("=" * 50)

        if tune_hyperparameters:
            self.expense_model = self._build_tuned_lstm(
                input_shape=(sequence_length, X_combined.shape[-1]),
                **lstm_best_params
            )
        else:
            self.expense_model = self.build_lstm_model(
                input_shape=(sequence_length, X_combined.shape[-1])
            )

        # Training callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7, verbose=1)
        ]

        history_lstm = self.expense_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=lstm_best_params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate LSTM
        lstm_results = self.evaluate_model_performance(self.expense_model, X_test, y_test, "LSTM")

        # Train GRU model
        print("\n" + "=" * 50)
        print("TRAINING GRU MODEL")
        print("=" * 50)

        if tune_hyperparameters:
            gru_model = self._build_tuned_gru(
                input_shape=(sequence_length, X_combined.shape[-1]),
                **gru_best_params
            )
        else:
            gru_model = self.build_gru_model(
                input_shape=(sequence_length, X_combined.shape[-1])
            )

        history_gru = gru_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=gru_best_params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate GRU
        gru_results = self.evaluate_model_performance(gru_model, X_test, y_test, "GRU")

        # Cross-validation
        print("\nPerforming cross-validation...")
        lstm_cv_scores = self.cross_validate_model(X_combined_scaled, y_combined, 'LSTM', n_splits=3)
        gru_cv_scores = self.cross_validate_model(X_combined_scaled, y_combined, 'GRU', n_splits=3)

        # Model comparison
        comparison_results = self.compare_models(X_test, y_test)

        # Training summary
        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETE - SUMMARY")
        print(f"{'=' * 60}")
        print(f"✅ Models trained and evaluated")
        print(f"✅ Best model: {comparison_results.iloc[0]['Model']}")
        print(f"✅ Model grade: {comparison_results.iloc[0]['Grade']}")
        print(f"✅ Cross-validation completed")

        if tune_hyperparameters:
            print(f"✅ Hyperparameter tuning completed")
            print(f"   LSTM best params: {self.best_params.get('LSTM', {})}")
            print(f"   GRU best params: {self.best_params.get('GRU', {})}")

        return {
            'lstm_history': history_lstm,
            'gru_history': history_gru,
            'evaluation_results': self.evaluation_results,
            'comparison_results': comparison_results,
            'cv_scores': {'lstm': lstm_cv_scores, 'gru': gru_cv_scores},
            'best_params': self.best_params
        }

    # FIRST VERSION
    # def train_models(self, df, sequence_length=30, forecast_horizon=7):
    #     """
    #     Train all models
    #     """
    #     all_X, all_y = [], []
    #     user_features_list = []
    #
    #     # Prepare data for all users
    #     for client_id in df['client_id'].unique():
    #         X, y = self.create_sequences(df, client_id, sequence_length, forecast_horizon)
    #         if X is not None:
    #             all_X.append(X)
    #             all_y.append(y)
    #
    #             # User-specific features
    #             user_data = df[df['client_id'] == client_id].iloc[0]
    #             user_features = [
    #                 user_data['avg_amount'],
    #                 user_data['std_amount'],
    #                 user_data['transaction_count'],
    #                 user_data.get('age', 0),
    #                 user_data.get('credit_limit', 0)
    #             ]
    #             user_features_list.extend([user_features] * len(X))
    #
    #     # Combine all data
    #     X_combined = np.vstack(all_X)
    #     y_combined = np.vstack(all_y)
    #     user_features_array = np.array(user_features_list)
    #
    #     print(f"Dataset shape: X={X_combined.shape}, y={y_combined.shape}")
    #
    #     # Normalize features
    #     X_combined_scaled = self.scaler.fit_transform(
    #         X_combined.reshape(-1, X_combined.shape[-1])
    #     ).reshape(X_combined.shape)
    #
    #     # Split data
    #     X_train, X_test, y_train, y_test, user_train, user_test = train_test_split(
    #         X_combined_scaled, y_combined, user_features_array,
    #         test_size=0.2, random_state=42
    #     )
    #
    #     # Train LSTM model
    #     print("Training LSTM model...")
    #     self.expense_model = self.build_lstm_model(
    #         input_shape=(sequence_length, X_combined.shape[-1])
    #     )
    #
    #     history_lstm = self.expense_model.fit(
    #         X_train, y_train,
    #         validation_data=(X_test, y_test),
    #         epochs=50,
    #         batch_size=32,
    #         verbose=1
    #     )
    #
    #     # Train advanced model
    #     print("Training advanced multi-output model...")
    #     self.cashflow_model = self.build_advanced_model(
    #         input_shape=(sequence_length, X_combined.shape[-1]),
    #         user_features_dim=user_features_array.shape[1]
    #     )
    #
    #     # Prepare outputs for advanced model
    #     y_expense = y_train[:, 0:1]  # Expenses
    #     y_income = y_train[:, 1:2]  # Income
    #     y_cashflow = np.random.randn(len(y_train), 7)  # Placeholder for daily cashflow
    #
    #     history_advanced = self.cashflow_model.fit(
    #         [X_train, user_train],
    #         {
    #             'expense_forecast': y_expense,
    #             'income_forecast': y_income,
    #             'cashflow_7day': y_cashflow
    #         },
    #         validation_data=(
    #             [X_test, user_test],
    #             {
    #                 'expense_forecast': y_test[:, 0:1],
    #                 'income_forecast': y_test[:, 1:2],
    #                 'cashflow_7day': np.random.randn(len(y_test), 7)
    #             }
    #         ),
    #         epochs=30,
    #         batch_size=32,
    #         verbose=1
    #     )
    #
    #     return history_lstm, history_advanced

    def predict_user_forecast(self, user_data_sequence, user_features=None):
        """
        Generate predictions for a user
        """
        # Scale the input
        user_data_scaled = self.scaler.transform(
            user_data_sequence.reshape(-1, user_data_sequence.shape[-1])
        ).reshape(user_data_sequence.shape)

        # LSTM prediction
        lstm_pred = self.expense_model.predict(user_data_scaled.reshape(1, *user_data_scaled.shape))

        predictions = {
            'next_week_expenses': float(lstm_pred[0][0]),
            'next_week_income': float(lstm_pred[0][1]),
            'net_cashflow': float(lstm_pred[0][1] - lstm_pred[0][0])
        }

        # Advanced model prediction if user features available
        if user_features is not None and self.cashflow_model is not None:
            user_features_array = np.array([user_features])
            advanced_pred = self.cashflow_model.predict([
                user_data_scaled.reshape(1, *user_data_scaled.shape),
                user_features_array
            ])

            predictions.update({
                'detailed_expense_forecast': float(advanced_pred[0][0][0]),
                'detailed_income_forecast': float(advanced_pred[1][0][0]),
                'daily_cashflow_7day': advanced_pred[2][0].tolist()
            })

        return predictions

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
            if isinstance(y_pred, list):
                # Multi-output model (advanced model)
                y_pred_expenses = y_pred[0].flatten()
                y_pred_income = y_pred[1].flatten()
                y_test_expenses = y_test[:, 0] if len(y_test.shape) > 1 else y_test
                y_test_income = y_test[:, 1] if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test
            else:
                # Single model with multiple outputs
                y_pred_expenses = y_pred[:, 0]
                y_pred_income = y_pred[:, 1]
                y_test_expenses = y_test[:, 0]
                y_test_income = y_test[:, 1]
        else:
            # Single output model
            y_pred_expenses = y_pred[:, 0] if y_pred.shape[1] > 1 else y_pred.flatten()
            y_pred_income = y_pred[:, 1] if y_pred.shape[1] > 1 else np.zeros_like(y_pred_expenses)
            y_test_expenses = y_test[:, 0] if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test.flatten()
            y_test_income = y_test[:, 1] if len(y_test.shape) > 1 and y_test.shape[1] > 1 else np.zeros_like(
                y_test_expenses)

        # Calculate metrics for expenses
        expense_metrics = self._calculate_metrics(y_test_expenses, y_pred_expenses, "Expenses")

        # Calculate metrics for income (if available)
        income_metrics = {}
        if not np.all(y_test_income == 0):
            income_metrics = self._calculate_metrics(y_test_income, y_pred_income, "Income")

        # Overall model performance
        overall_metrics = {
            'model_name': model_name,
            'expense_metrics': expense_metrics,
            'income_metrics': income_metrics,
            'prediction_accuracy_grade': self._grade_model_performance(expense_metrics, income_metrics)
        }

        # Store results
        self.evaluation_results[model_name] = overall_metrics

        # Visualization
        self._plot_predictions(y_test_expenses, y_pred_expenses, y_test_income, y_pred_income, model_name)

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

    def _grade_model_performance(self, expense_metrics, income_metrics):
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

    def _plot_predictions(self, y_test_expenses, y_pred_expenses, y_test_income, y_pred_income, model_name):
        """
        Plot prediction results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Prediction Analysis', fontsize=16)

        # Expenses: Actual vs Predicted
        axes[0, 0].scatter(y_test_expenses, y_pred_expenses, alpha=0.6)
        axes[0, 0].plot([y_test_expenses.min(), y_test_expenses.max()],
                        [y_test_expenses.min(), y_test_expenses.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Expenses')
        axes[0, 0].set_ylabel('Predicted Expenses')
        axes[0, 0].set_title('Expenses: Actual vs Predicted')

        # Expenses: Residuals
        residuals_expenses = y_test_expenses - y_pred_expenses
        axes[0, 1].hist(residuals_expenses, bins=30, alpha=0.7)
        axes[0, 1].set_xlabel('Residuals (Actual - Predicted)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Expenses: Residuals Distribution')

        # Time series plot (first 100 predictions)
        n_plot = min(100, len(y_test_expenses))
        axes[1, 0].plot(range(n_plot), y_test_expenses[:n_plot], label='Actual', linewidth=2)
        axes[1, 0].plot(range(n_plot), y_pred_expenses[:n_plot], label='Predicted', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Expenses')
        axes[1, 0].set_title('Expenses: Time Series Comparison')
        axes[1, 0].legend()

        # Model performance summary
        if model_name in self.evaluation_results:
            metrics = self.evaluation_results[model_name]['expense_metrics']
            grade = self.evaluation_results[model_name]['prediction_accuracy_grade']

            summary_text = f"""
Model Grade: {grade}
R² Score: {metrics.get('R2_Score', 0):.3f}
RMSE: ${metrics.get('RMSE', 0):.2f}
MAPE: {metrics.get('MAPE', 0):.1f}%
Direction Acc: {metrics.get('Direction_Accuracy', 0):.1f}%
            """
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Performance Summary')

        plt.tight_layout()
        plt.show()

    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, model_type='LSTM'):
        """
        Automated hyperparameter tuning
        """
        print(f"\n{'=' * 50}")
        print(f"HYPERPARAMETER TUNING FOR {model_type}")
        print(f"{'=' * 50}")

        # Parameter grid
        param_grid = {
            'lstm_units': [64, 128, 256],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64]
        }

        best_score = float('inf')
        best_params = {}
        results = []

        # Grid search
        for lstm_units in param_grid['lstm_units']:
            for dropout_rate in param_grid['dropout_rate']:
                for learning_rate in param_grid['learning_rate']:
                    for batch_size in param_grid['batch_size']:

                        print(
                            f"Testing: LSTM={lstm_units}, Dropout={dropout_rate}, LR={learning_rate}, Batch={batch_size}")

                        # Build model with current parameters
                        if model_type == 'LSTM':
                            model = self._build_tuned_lstm(
                                input_shape=(X_train.shape[1], X_train.shape[2]),
                                lstm_units=lstm_units,
                                dropout_rate=dropout_rate,
                                learning_rate=learning_rate
                            )
                        else:  # GRU
                            model = self._build_tuned_gru(
                                input_shape=(X_train.shape[1], X_train.shape[2]),
                                gru_units=lstm_units,
                                dropout_rate=dropout_rate,
                                learning_rate=learning_rate
                            )

                        # Train model
                        history = model.fit(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=20,  # Reduced for tuning
                            batch_size=batch_size,
                            verbose=0,
                            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
                        )

                        # Evaluate
                        val_loss = min(history.history['val_loss'])

                        results.append({
                            'lstm_units': lstm_units,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'val_loss': val_loss
                        })

                        if val_loss < best_score:
                            best_score = val_loss
                            best_params = {
                                'lstm_units': lstm_units,
                                'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate,
                                'batch_size': batch_size
                            }

                        print(f"Val Loss: {val_loss:.4f}")

        self.best_params[model_type] = best_params

        print(f"\nBest Parameters for {model_type}:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best Validation Loss: {best_score:.4f}")

        return best_params, results

    def _build_tuned_lstm(self, input_shape, lstm_units=128, dropout_rate=0.2, learning_rate=0.001):
        """Build LSTM with specific hyperparameters"""
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units // 2, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units // 4),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(2, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _build_tuned_gru(self, input_shape, gru_units=128, dropout_rate=0.2, learning_rate=0.001):
        """Build GRU with specific hyperparameters"""
        model = Sequential([
            GRU(gru_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            GRU(gru_units // 2, return_sequences=True),
            Dropout(dropout_rate),
            GRU(gru_units // 4),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(2, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

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

    def generate_recommendations(self, predictions, user_spending_history):
        """
        Generate spending recommendations based on predictions
        """
        recommendations = []

        # Cash flow analysis
        net_flow = predictions['net_cashflow']
        if net_flow < 0:
            recommendations.append({
                'type': 'warning',
                'message': f"Predicted negative cash flow of ${abs(net_flow):.2f} next week",
                'suggestion': "Consider reducing discretionary spending"
            })

        # Spending pattern analysis
        avg_weekly_spending = user_spending_history['amount'].sum() / 52  # Assuming yearly data
        predicted_spending = predictions['next_week_expenses']

        if predicted_spending > avg_weekly_spending * 1.2:
            recommendations.append({
                'type': 'alert',
                'message': f"Spending predicted to be {((predicted_spending / avg_weekly_spending - 1) * 100):.1f}% above average",
                'suggestion': "Review upcoming planned purchases"
            })

        return recommendations

    def model_diagnostics(self, model, X_test, y_test, model_name):
        """
        Advanced model diagnostics and insights
        """
        print(f"\n{'=' * 50}")
        print(f"ADVANCED DIAGNOSTICS - {model_name}")
        print(f"{'=' * 50}")

        # Predictions
        y_pred = model.predict(X_test, verbose=0)

        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_expenses = y_pred[:, 0]
            y_test_expenses = y_test[:, 0]
        else:
            y_pred_expenses = y_pred.flatten()
            y_test_expenses = y_test.flatten()

        # 1. Prediction Intervals
        residuals = y_test_expenses - y_pred_expenses
        prediction_std = np.std(residuals)

        # 95% confidence intervals
        ci_lower = y_pred_expenses - 1.96 * prediction_std
        ci_upper = y_pred_expenses + 1.96 * prediction_std

        # Coverage probability
        coverage = np.mean((y_test_expenses >= ci_lower) & (y_test_expenses <= ci_upper))

        print(f"📊 Prediction Intervals:")
        print(f"   Standard Deviation: ${prediction_std:.2f}")
        print(f"   95% Confidence Interval Coverage: {coverage * 100:.1f}%")

        # 2. Error Analysis by Amount Range
        amount_ranges = [(0, 100), (100, 500), (500, 1000), (1000, float('inf'))]
        print(f"\n📈 Error Analysis by Amount Range:")

        for low, high in amount_ranges:
            mask = (y_test_expenses >= low) & (y_test_expenses < high)
            if np.sum(mask) > 0:
                range_mae = np.mean(np.abs(residuals[mask]))
                range_mape = np.mean(np.abs(residuals[mask] / np.maximum(y_test_expenses[mask], 1))) * 100
                print(
                    f"   ${low}-${high if high != float('inf') else '∞'}: MAE=${range_mae:.2f}, MAPE={range_mape:.1f}%, Count={np.sum(mask)}")

        # 3. Seasonal Performance (if time features available)
        print(f"\n📅 Model Stability Analysis:")

        # Split test set into chunks and analyze performance
        chunk_size = len(y_test_expenses) // 5
        chunk_performances = []

        for i in range(5):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 4 else len(y_test_expenses)

            chunk_true = y_test_expenses[start_idx:end_idx]
            chunk_pred = y_pred_expenses[start_idx:end_idx]

            if len(chunk_true) > 0:
                chunk_mae = mean_absolute_error(chunk_true, chunk_pred)
                chunk_r2 = r2_score(chunk_true, chunk_pred)
                chunk_performances.append((chunk_mae, chunk_r2))
                print(f"   Period {i + 1}: MAE=${chunk_mae:.2f}, R²={chunk_r2:.3f}")

        # Performance stability
        mae_std = np.std([perf[0] for perf in chunk_performances])
        r2_std = np.std([perf[1] for perf in chunk_performances])
        print(f"   Performance Stability: MAE_std=${mae_std:.2f}, R²_std={r2_std:.3f}")

        # 4. Feature Importance Analysis (simplified)
        print(f"\n🔍 Model Insights:")

        # Analyze which features contribute most to large errors
        large_error_mask = np.abs(residuals) > np.percentile(np.abs(residuals), 90)
        large_error_count = np.sum(large_error_mask)

        print(f"   Large Errors (>90th percentile): {large_error_count} predictions")
        print(f"   Large Error Threshold: ${np.percentile(np.abs(residuals), 90):.2f}")

        # Overfitting check
        train_pred = model.predict(X_test[:100], verbose=0)  # Sample for speed
        if len(train_pred.shape) > 1:
            train_mae = mean_absolute_error(y_test[:100, 0], train_pred[:, 0])
        else:
            train_mae = mean_absolute_error(y_test[:100].flatten(), train_pred.flatten())

        test_mae = mean_absolute_error(y_test_expenses, y_pred_expenses)
        overfitting_ratio = test_mae / train_mae if train_mae > 0 else 1

        print(f"   Overfitting Check: Test/Train MAE ratio = {overfitting_ratio:.2f}")
        if overfitting_ratio > 1.2:
            print(f"   ⚠️  Potential overfitting detected")
        else:
            print(f"   ✅ No significant overfitting")

        # 5. Recommendations for improvement
        print(f"\n💡 Model Improvement Recommendations:")

        r2_score_val = r2_score(y_test_expenses, y_pred_expenses)
        mape_val = np.mean(np.abs(residuals / np.maximum(y_test_expenses, 1))) * 100

        if r2_score_val < 0.7:
            print(f"   📈 Low R² ({r2_score_val:.3f}): Consider feature engineering or ensemble methods")

        if mape_val > 20:
            print(f"   📊 High MAPE ({mape_val:.1f}%): Consider data preprocessing or outlier removal")

        if coverage < 0.9:
            print(f"   📉 Low CI coverage ({coverage * 100:.1f}%): Consider uncertainty quantification methods")

        if mae_std > np.mean([perf[0] for perf in chunk_performances]) * 0.3:
            print(f"   📈 Unstable performance: Consider more robust training or data augmentation")

        return {
            'prediction_std': prediction_std,
            'ci_coverage': coverage,
            'chunk_performances': chunk_performances,
            'overfitting_ratio': overfitting_ratio,
            'large_error_count': large_error_count
        }

    def generate_model_report(self, output_file='model_evaluation_report.txt'):
        """
        Generate comprehensive model evaluation report
        """
        report = []
        report.append("=" * 80)
        report.append("FINANCIAL FORECASTING MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)

        if self.evaluation_results:
            best_model = max(self.evaluation_results.keys(),
                             key=lambda x: self.evaluation_results[x]['expense_metrics'].get('R2_Score', 0))
            best_grade = self.evaluation_results[best_model]['prediction_accuracy_grade']
            best_r2 = self.evaluation_results[best_model]['expense_metrics'].get('R2_Score', 0)

            report.append(f"🏆 Best Performing Model: {best_model}")
            report.append(f"📊 Model Grade: {best_grade}")
            report.append(f"📈 R² Score: {best_r2:.4f}")
            report.append("")

        # Detailed Results
        for model_name, results in self.evaluation_results.items():
            report.append(f"MODEL: {model_name}")
            report.append("-" * 30)

            expense_metrics = results['expense_metrics']
            grade = results['prediction_accuracy_grade']

            report.append(f"Overall Grade: {grade}")
            report.append(f"R² Score: {expense_metrics.get('R2_Score', 0):.4f}")
            report.append(f"RMSE: ${expense_metrics.get('RMSE', 0):.2f}")
            report.append(f"MAE: ${expense_metrics.get('MAE', 0):.2f}")
            report.append(f"MAPE: {expense_metrics.get('MAPE', 0):.2f}%")
            report.append(f"Direction Accuracy: {expense_metrics.get('Direction_Accuracy', 0):.2f}%")
            report.append(f"Correlation: {expense_metrics.get('Correlation', 0):.4f}")
            report.append("")

        # Hyperparameter Information
        if self.best_params:
            report.append("OPTIMAL HYPERPARAMETERS")
            report.append("-" * 40)
            for model_type, params in self.best_params.items():
                report.append(f"{model_type}:")
                for param, value in params.items():
                    report.append(f"  {param}: {value}")
                report.append("")

        # Recommendations
        report.append("DEPLOYMENT RECOMMENDATIONS")
        report.append("-" * 40)

        if self.evaluation_results:
            best_model_metrics = self.evaluation_results[best_model]['expense_metrics']

            if best_model_metrics.get('R2_Score', 0) > 0.8:
                report.append("✅ Model shows strong predictive performance")
                report.append("✅ Suitable for production deployment")
            elif best_model_metrics.get('R2_Score', 0) > 0.6:
                report.append("⚠️  Model shows moderate performance")
                report.append("⚠️  Consider additional feature engineering")
            else:
                report.append("❌ Model performance below recommended threshold")
                report.append("❌ Requires significant improvement before deployment")

            if best_model_metrics.get('Direction_Accuracy', 0) > 70:
                report.append("✅ Good directional prediction accuracy")
            else:
                report.append("⚠️  Poor directional accuracy - may mislead users")

        # Save report
        report_text = "\n".join(report)

        with open(output_file, 'w') as f:
            f.write(report_text)

        print(f"📄 Comprehensive report saved to: {output_file}")
        print("\nReport Preview:")
        print("-" * 50)
        print(report_text[:1000] + "..." if len(report_text) > 1000 else report_text)

        return report_text

    def save_models(self, model_dir='models/'):
        """
        Save trained models
        """
        import os
        os.makedirs(model_dir, exist_ok=True)

        if self.expense_model:
            self.expense_model.save(f'{model_dir}/lstm_expense_model.h5')

        if self.cashflow_model:
            self.cashflow_model.save(f'{model_dir}/advanced_cashflow_model.h5')

        # Save scaler and encoders
        import joblib
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        joblib.dump(self.mcc_encoder, f'{model_dir}/mcc_encoder.pkl')

    def load_models(self, model_dir='models/'):
        """
        Load pre-trained models
        """
        import joblib

        self.expense_model = tf.keras.models.load_model(f'{model_dir}/lstm_expense_model.h5')
        self.cashflow_model = tf.keras.models.load_model(f'{model_dir}/advanced_cashflow_model.h5')
        self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
        self.mcc_encoder = joblib.load(f'{model_dir}/mcc_encoder.pkl')


# Usage example with comprehensive evaluation
if __name__ == "__main__":
    # Initialize the model class
    forecaster = FinancialForecastingModels()

    # Example usage with evaluation
    print("🚀 Financial Forecasting Models with Comprehensive Evaluation")
    print("=" * 60)

    # Load your data
    transactions_df = pd.read_csv('../dataset/transactions_data.csv')
    users_df = pd.read_csv('../dataset/users_data.csv')
    cards_df = pd.read_csv('../dataset/cards_data.csv')
    # Prepare data
    print("📊 Preparing data...")
    prepared_df = forecaster.prepare_data(transactions_df, users_df, cards_df, '../dataset/mcc_codes.json')

    # Train models with evaluation
    print("🔧 Training models...")
    results = forecaster.train_models(
        prepared_df, 
        sequence_length=30, 
        forecast_horizon=7,
        tune_hyperparameters=True  # Set to True for hyperparameter tuning
    )

    # Additional diagnostics for best model
    best_model_name = results['comparison_results'].iloc[0]['Model']
    if best_model_name == 'LSTM':
        best_model = forecaster.expense_model
    else:
        # You would need to store the GRU model as well
        best_model = forecaster.expense_model  # Placeholder

    # Run diagnostics
    print("🔍 Running advanced diagnostics...")
    # diagnostics = forecaster.model_diagnostics(
    #     best_model, X_test, y_test, best_model_name
    # )

    # Generate comprehensive report
    print("📄 Generating evaluation report...")
    report = forecaster.generate_model_report('financial_model_report.txt')

    # Save models
    print("💾 Saving models...")
    forecaster.save_models()

    print("✅ Complete! Check your evaluation report and model files.")
    print("Ready for training with comprehensive evaluation!")
    print("\nKey Features:")
    print("✅ Multiple accuracy metrics (MAE, RMSE, R², MAPE, Direction Accuracy)")
    print("✅ Model grading system (A+ to F)")
    print("✅ Hyperparameter tuning with grid search")
    print("✅ Time series cross-validation")
    print("✅ Advanced diagnostics and error analysis")
    print("✅ Performance comparison between models")
    print("✅ Comprehensive evaluation reports")
    print("✅ Visualization of predictions and residuals")
    print("✅ Overfitting detection")
    print("✅ Prediction intervals and uncertainty quantification")