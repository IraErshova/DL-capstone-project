import os
from flask import Blueprint, render_template, request, jsonify, current_app
import pandas as pd
from data.processor import process_transactions

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
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400 