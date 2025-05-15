from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import logging
from datetime import datetime
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model artifacts
try:
    os.makedirs('models', exist_ok=True)
    model_pipeline = joblib.load('../model/random_forest_model.pkl')
    logger.info("Model pipeline loaded successfully")
    
    with open('../model/feature_names.pkl', 'rb') as f:
        expected_features = joblib.load(f)
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise e

def preprocess_input(form_data):
    """Transform form input into model-ready format."""
    try:
        df = pd.DataFrame([form_data])
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['Month'] = df['Date'].dt.month
        df['weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x > 5 else 0)
        df['weekdays'] = df['DayOfWeek'].apply(lambda x: 1 if x <= 5 else 0)
        df.drop('Date', axis=1, inplace=True)

        # Convert StoreType and Assortment to one-hot
        store_type = form_data.get('StoreType', 'a')
        df['StoreType_b'] = 1 if store_type == 'b' else 0
        df['StoreType_c'] = 1 if store_type == 'c' else 0
        df['StoreType_d'] = 1 if store_type == 'd' else 0

        assortment = form_data.get('Assortment', 'a')
        df['Assortment_b'] = 1 if assortment == 'b' else 0
        df['Assortment_c'] = 1 if assortment == 'c' else 0

        # Ensure all expected features exist
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        df = df[expected_features]
        return df
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise ValueError(f"Input processing failed: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            if request.is_json:
                form_data = request.get_json()
            else:
                form_data = request.form.to_dict()
            
            # Debug: Print received data
            logger.info(f"Received form data: {form_data}")
            
            # Convert numeric fields - ensure all required fields are included
            numeric_fields = ['Store', 'Customers', 'Open', 'Promo', 'SchoolHoliday',
                            'CompetitionDistance', 'Promo2', 'Sales_lag_1', 'Sales_lag_7',
                            'DaysToNextHoliday', 'DaysAfterLastHoliday']
            
            for field in numeric_fields:
                if field in form_data:
                    try:
                        form_data[field] = float(form_data[field])
                    except (ValueError, TypeError):
                        raise ValueError(f"Invalid value for {field}. Must be a number.")
            
            # Ensure required fields are present
            required_fields = ['Date', 'Store', 'StoreType', 'Assortment']
            for field in required_fields:
                if field not in form_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Preprocess and predict
            processed_data = preprocess_input(form_data)
            prediction = model_pipeline.predict(processed_data)[0]
            
            return jsonify({
                'success': True,
                'prediction': f"${prediction:,.2f}"
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)