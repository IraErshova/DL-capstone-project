import os
from flask import Blueprint, render_template, request, jsonify, current_app
import pandas as pd
from data.processor import process_transactions
from model.forecast import FinancialForecastingModels
from model.preprocessing_data import create_sequences_improved

main = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_filename(filename):
    """Basic filename sanitization"""
    # Remove any directory components
    filename = os.path.basename(filename)
    # Replace any non-alphanumeric characters with underscore
    filename = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in filename)
    return filename

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Process the file
        try:
            df = pd.read_csv(filepath)
            results = process_transactions(df)
            print(results)
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400

@main.route('/predict', methods=['POST'])
def predict_expenses():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read the uploaded file
            df = pd.read_csv(file)
            
            # Get client_id from request or use the first one in the dataset
            client_id = request.form.get('client_id', df['client_id'].iloc[0])
            
            # Create sequences for prediction
            X, y, user_features = create_sequences_improved(df, client_id)
            
            if X is None:
                return jsonify({
                    'error': 'Not enough data for prediction. Need at least 30 days of transaction history.'
                }), 400
            
            # Initialize forecaster and make prediction
            forecaster = FinancialForecastingModels()
            prediction = forecaster.predict_user_forecast(X[-1:], user_features[-1:] if user_features is not None else None)
            
            # Convert any remaining NumPy types to Python native types
            response = {
                'prediction': {
                    'next_week_expenses': float(prediction['next_week_expenses']),
                },
                'client_id': int(client_id)
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400 