# Data Preprocessing and Normalization
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def clean_currency_columns(df):
    """
    Clean currency columns that contain '$' and convert to float
    """
    currency_columns = ['per_capita_income', 'yearly_income', 'total_debt', 'credit_limit']

    for col in currency_columns:
        if col in df.columns:
            # Remove '$' and ',' then convert to float
            df[col] = (df[col]
                       .astype(str)
                       .str.replace('$', '', regex=False)
                       .str.replace(',', '', regex=False)
                       .replace('nan', np.nan)
                       .replace('None', np.nan))

            # Convert to numeric, errors='coerce' will convert invalid parsing to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def prepare_user_features(df):
    """
    Prepare and normalize user features properly
    """
    # Clean currency columns first
    df = clean_currency_columns(df)

    # Fill missing values with median for numerical columns
    numerical_cols = ['current_age', 'credit_score', 'num_credit_cards',
                      'per_capita_income', 'yearly_income', 'total_debt',
                      'credit_limit', 'latitude', 'longitude']

    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical variables
    categorical_encoders = {}
    categorical_cols = ['use_chip', 'gender', 'card_brand', 'card_type',
                        'has_chip', 'card_on_dark_web']

    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col + '_encoded'] = encoder.fit_transform(df[col].fillna('unknown'))
            categorical_encoders[col] = encoder

    return df, categorical_encoders


def create_user_feature_vector(user_data):
    """
    Create a proper user feature vector with all relevant features
    """
    def clean_currency(value):
        if isinstance(value, str):
            # Remove '$' and ',' from currency values
            return float(value.replace('$', '').replace(',', ''))
        return float(value)

    features = []

    # Spending patterns (already calculated)
    features.extend([
        float(user_data.get('avg_amount', 0)),
        float(user_data.get('std_amount', 0)),
        float(user_data.get('transaction_count', 0)),
        clean_currency(user_data.get('total_spent', 0))
    ])

    # Demographics
    features.extend([
        float(user_data.get('current_age', 0)),
        float(user_data.get('credit_score', 0)),
        float(user_data.get('num_credit_cards', 0)),
        clean_currency(user_data.get('per_capita_income', 0)),
        clean_currency(user_data.get('yearly_income', 0)),
        clean_currency(user_data.get('total_debt', 0)),
        clean_currency(user_data.get('credit_limit', 0))
    ])

    # Location features
    features.extend([
        float(user_data.get('latitude', 0)),
        float(user_data.get('longitude', 0))
    ])

    # Encoded categorical features
    features.extend([
        float(user_data.get('use_chip_encoded', 0)),
        float(user_data.get('gender_encoded', 0)),
        float(user_data.get('card_brand_encoded', 0)),
        float(user_data.get('card_type_encoded', 0)),
        float(user_data.get('has_chip_encoded', 0)),
        float(user_data.get('card_on_dark_web_encoded', 0))
    ])

    return np.array(features, dtype=np.float32)


def improved_data_preprocessing():
    """
    Complete improved data preprocessing pipeline
    """
    # Load data
    transactions_df = pd.read_csv('dataset/transactions_data.csv')
    transactions_df = transactions_df.sample(frac=0.01, random_state=42).reset_index(drop=True)
    users_df = pd.read_csv('dataset/users_data.csv')
    cards_df = pd.read_csv('dataset/cards_data.csv')

    # Load MCC codes
    with open('dataset/mcc_codes.json', 'r') as f:
        mcc_codes = json.load(f)

    # Merge dataframes
    df = transactions_df.merge(users_df, left_on='client_id', right_on='id',
                               how='left', suffixes=('', '_user'))
    df = df.merge(cards_df, left_on='card_id', right_on='id',
                  how='left', suffixes=('', '_card'))

    # Drop unwanted columns
    df = df.drop(columns=['id_user', 'id_card', 'client_id_card'], errors='ignore')

    # Clean and preprocess data
    df, categorical_encoders = prepare_user_features(df)

    # Feature engineering
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Clean amount column
    df['amount'] = (df['amount']
                    .astype(str)
                    .str.replace('$', '', regex=False)
                    .astype(float)
                    .apply(lambda x: abs(x)))

    # MCC encoding
    mcc_encoder = LabelEncoder()
    df['mcc_encoded'] = mcc_encoder.fit_transform(df['mcc'].fillna('unknown'))

    # User spending patterns
    user_stats = df.groupby('client_id').agg({
        'amount': ['mean', 'std', 'sum'],
        'date': ['count']
    }).reset_index()
    user_stats.columns = ['client_id', 'avg_amount', 'std_amount', 'total_spent', 'transaction_count']

    df = df.merge(user_stats, on='client_id', how='left')

    return df, mcc_encoder, categorical_encoders


def create_sequences_improved(df, client_id, sequence_length=30, forecast_horizon=7):
    """
    Improved sequence creation with proper feature handling
    """
    user_data = df[df['client_id'] == client_id].sort_values('date')

    if len(user_data) < sequence_length + forecast_horizon:
        return None, None, None

    # Features for sequence input (normalized)
    sequence_features = [
        'amount', 'mcc_encoded', 'hour', 'day_of_week',
        'month', 'is_weekend'
    ]

    # User features (to be normalized separately)
    user_feature_names = [
        'avg_amount', 'std_amount', 'transaction_count', 'total_spent',
        'current_age', 'credit_score', 'num_credit_cards',
        'per_capita_income', 'yearly_income', 'total_debt', 'credit_limit',
        'latitude', 'longitude',
        'use_chip_encoded', 'gender_encoded', 'card_brand_encoded',
        'card_type_encoded', 'has_chip_encoded', 'card_on_dark_web_encoded'
    ]

    X, y, user_features = [], [], []

    for i in range(len(user_data) - sequence_length - forecast_horizon + 1):
        # Sequence data
        seq_data = user_data.iloc[i:i + sequence_length][sequence_features].values
        X.append(seq_data)

        # Target
        future_data = user_data.iloc[i + sequence_length:i + sequence_length + forecast_horizon]
        future_expenses = future_data['amount'].sum()
        y.append([future_expenses])

        # User features (same for all sequences of this user)
        user_row = user_data.iloc[i]
        user_feat = create_user_feature_vector(user_row)
        user_features.append(user_feat)

    return np.array(X), np.array(y), np.array(user_features)


def normalize_all_features(X_combined, user_features_array):
    """
    Properly normalize both sequence and user features
    """
    print("Normalizing sequence features...")
    # Normalize sequence features
    n_samples, seq_len, n_features = X_combined.shape
    X_reshaped = X_combined.reshape(-1, n_features)

    sequence_scaler = StandardScaler()
    X_scaled = sequence_scaler.fit_transform(X_reshaped)
    X_combined_scaled = X_scaled.reshape(n_samples, seq_len, n_features)

    print("Normalizing user features...")
    # Normalize user features
    user_scaler = StandardScaler()
    user_features_scaled = user_scaler.fit_transform(user_features_array)

    return X_combined_scaled, user_features_scaled, sequence_scaler, user_scaler

def main_preprocessing_pipeline():
    """
    Main preprocessing pipeline
    """
    # Step 1: Load and preprocess data
    df, mcc_encoder, categorical_encoders = improved_data_preprocessing()

    # Step 2: Create sequences for all users
    client_ids = df['client_id'].unique()
    all_X, all_y, all_user_features = [], [], []

    print(f"Processing {len(client_ids)} unique clients...")

    for client_id in tqdm(client_ids, desc="Creating sequences"):
        X, y, user_feat = create_sequences_improved(df, client_id)
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            all_user_features.append(user_feat)

    # Step 3: Combine all data
    X_combined = np.vstack(all_X)
    y_combined = np.vstack(all_y)
    user_features_combined = np.vstack(all_user_features)

    # Step 4: Normalize features properly
    X_scaled, user_features_scaled, seq_scaler, user_scaler = normalize_all_features(
        X_combined, user_features_combined
    )

    # Step 5: Split data
    indices = np.arange(len(X_scaled))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    # Create splits
    X_train, X_val, X_test = X_scaled[train_idx], X_scaled[val_idx], X_scaled[test_idx]
    y_train, y_val, y_test = y_combined[train_idx], y_combined[val_idx], y_combined[test_idx]
    user_train = user_features_scaled[train_idx]
    user_val = user_features_scaled[val_idx]
    user_test = user_features_scaled[test_idx]

    # Save scalers and encoders
    scalers_and_encoders = {
        'sequence_scaler': seq_scaler,
        'user_scaler': user_scaler,
        'mcc_encoder': mcc_encoder,
        'categorical_encoders': categorical_encoders
    }

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'user_train': user_train, 'user_val': user_val, 'user_test': user_test,
        'scalers_encoders': scalers_and_encoders
    }


# Additional utility functions
def save_preprocessing_artifacts(data_dict, save_dir='preprocessed_data/'):
    """
    Save all preprocessing artifacts
    """
    import os
    import joblib

    os.makedirs(save_dir, exist_ok=True)

    # Save arrays
    np.savez(f'{save_dir}/processed_data.npz',
             X_train=data_dict['X_train'],
             X_val=data_dict['X_val'],
             X_test=data_dict['X_test'],
             y_train=data_dict['y_train'],
             y_val=data_dict['y_val'],
             y_test=data_dict['y_test'],
             user_train=data_dict['user_train'],
             user_val=data_dict['user_val'],
             user_test=data_dict['user_test'])

    # Save scalers and encoders
    scalers = data_dict['scalers_encoders']
    for name, obj in scalers.items():
        joblib.dump(obj, f'{save_dir}/{name}.pkl')

    print(f"All preprocessing artifacts saved to {save_dir}")

def load_preprocessing_artifacts(save_dir='preprocessed_data/'):
    """
    Load all preprocessing artifacts
    """
    import joblib

    # Load arrays
    data = np.load(f'{save_dir}/processed_data.npz')

    # Load scalers and encoders
    scalers = {}
    scaler_files = ['sequence_scaler.pkl', 'user_scaler.pkl',
                    'mcc_encoder.pkl', 'categorical_encoders.pkl']

    for file in scaler_files:
        name = file.replace('.pkl', '')
        scalers[name] = joblib.load(f'{save_dir}/{file}')

    return {
        'X_train': data['X_train'], 'X_val': data['X_val'], 'X_test': data['X_test'],
        'y_train': data['y_train'], 'y_val': data['y_val'], 'y_test': data['y_test'],
        'user_train': data['user_train'], 'user_val': data['user_val'], 'user_test': data['user_test'],
        'scalers_encoders': scalers
    }