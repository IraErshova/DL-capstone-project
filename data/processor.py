import pandas as pd
import numpy as np

def process_transactions(df):
    """
    Process the uploaded transaction data and return basic analysis.
    This function processes transaction data for both analysis and prediction.
    """
    # Basic data validation
    required_columns = ['date', 'amount', 'mcc', 'client_id', 'card_id']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file must contain the following columns: date, amount, mcc, client_id, card_id")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Clean amount column
    df['amount'] = (df['amount']
                    .astype(str)
                    .str.replace('$', '', regex=False)
                    .astype(float)
                    .apply(lambda x: abs(x)))
    
    # Add time-based features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate user spending patterns
    user_stats = df.groupby('client_id').agg({
        'amount': ['mean', 'std', 'sum'],
        'date': ['count']
    }).reset_index()
    user_stats.columns = ['client_id', 'avg_amount', 'std_amount', 'total_spent', 'transaction_count']
    
    # Calculate monthly statistics
    monthly_stats = df.groupby(df['date'].dt.strftime('%Y-%m')).agg({
        'amount': ['sum', 'count']
    }).reset_index()
    monthly_stats.columns = ['month', 'total_amount', 'transaction_count']
    monthly_summary = monthly_stats.set_index('month').to_dict('index')
    
    # Basic analysis
    analysis = {
        'total_transactions': len(df),
        'total_amount': float(df['amount'].sum()),
        'avg_transaction': float(df['amount'].mean()),
        'std_transaction': float(df['amount'].std()),
        'mcc_distribution': df['mcc'].value_counts().to_dict(),
        'monthly_summary': monthly_summary,
        'client_summary': user_stats.to_dict('records'),
        'time_patterns': {
            'hourly_distribution': df.groupby('hour')['amount'].mean().to_dict(),
            'day_of_week_distribution': df.groupby('day_of_week')['amount'].mean().to_dict(),
            'monthly_distribution': df.groupby('month')['amount'].mean().to_dict(),
            'weekend_vs_weekday': {
                'weekend_avg': float(df[df['is_weekend'] == 1]['amount'].mean()),
                'weekday_avg': float(df[df['is_weekend'] == 0]['amount'].mean())
            }
        }
    }
    
    return analysis 