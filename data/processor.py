import pandas as pd
import numpy as np

def process_transactions(df):
    """
    Process the uploaded transaction data and return basic analysis.
    This is a placeholder function that you can expand based on your needs.
    """
    # Basic data validation
    required_columns = ['Date', 'Description', 'Amount', 'Category', 'Transaction Type']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file must contain the following columns: Date, Description, Amount, Category, Transaction Type")
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Basic analysis
    analysis = {
        'total_transactions': len(df),
        'total_income': float(df[df['Transaction Type'] == 'Income']['Amount'].sum()),
        'total_expenses': float(df[df['Transaction Type'] == 'Expense']['Amount'].sum()),
        'net_flow': float(df[df['Transaction Type'] == 'Income']['Amount'].sum() - 
                         df[df['Transaction Type'] == 'Expense']['Amount'].sum()),
        'categories': df['Category'].value_counts().to_dict(),
        'monthly_summary': df.groupby(df['Date'].dt.strftime('%Y-%m'))['Amount'].sum().to_dict()
    }
    
    return analysis 