# Financial Transaction Analysis and Forecasting

This application provides financial transaction analysis, forecasting, and recommendations based on your transaction history. Upload your CSV file with financial transactions, and the system will analyze your spending patterns, predict future expenses, and provide personalized recommendations.

## Features

- CSV file upload for transaction data
- Transaction analysis and visualization
- Expense and income forecasting
- Personalized financial recommendations
- Cash flow prediction

## Project Structure

```
├── app/                    # Web application
│   ├── static/            # CSS, JavaScript, and other static files
│   ├── templates/         # HTML templates
│   └── routes.py          # Web routes
├── model/                 # Machine learning model directory
│   ├── train.py          # Model training script
│   └── predict.py        # Prediction functions
├── data/                  # Data processing and utilities
│   └── processor.py      # Data processing functions
├── requirements.txt       # Python dependencies
└── config.py             # Configuration settings
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python run.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Data Format

The application expects a CSV file with the following columns:
- Date
- Description
- Amount
- Category
- Transaction Type (Income/Expense)

## Development

- The `model/` directory is where you'll develop and train your machine learning model
- Use the provided data processing utilities in `data/processor.py` to prepare your data
- The web interface is built with Flask and uses modern frontend technologies


## Dataset

https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets?select=transactions_data.csv

### Overview
This comprehensive financial dataset combines transaction records, customer information, and card data from a banking institution, spanning across the 2010s decade. The dataset is designed for multiple analytical purposes, including synthetic fraud detection, customer behavior analysis, and expense forecasting.

**Dataset Components**
1. Transaction Data (transactions_data.csv)
   Detailed transaction records including amounts, timestamps, and merchant details
   Covers transactions throughout the 2010s
   Features transaction types, amounts, and merchant information
   Perfect for analyzing spending patterns and building fraud detection models

2. Card Information (cards_dat.csv)
   Credit and debit card details
   Includes card limits, types, and activation dates
   Links to customer accounts via card_id
   Essential for understanding customer financial profiles

3. Merchant Category Codes (mcc_codes.json)
   Standard classification codes for business types
   Enables transaction categorization and spending analysis
   Industry-standard MCC codes with descriptions

4. Fraud Labels (train_fraud_labels.json)
   Binary classification labels for transactions
   Indicates fraudulent vs. legitimate transactions
   Ideal for training supervised fraud detection models

5. User Data (users_data)
   Demographic information about customers
   Account-related details
   Enables customer segmentation and personalized analysis
   Use Cases and Applications

