import pandas as pd
import logging
import joblib
import sys
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from io import StringIO
from fastapi import Response  # For debugging
import json  # For debugging JSON response



# Set up logging
logging.basicConfig(level=logging.INFO)

# Import your custom modules
sys.path.insert(0, '/10 A KAI 2/Week 4/Sales_Forcasting/')
from Scripts.Model_Preprocess import SalesForecasting
from Scripts.Model_build import ModelPrepro

# Initialize FastAPI app
app = FastAPI()

# Define the response model for the entire prediction output
class PredictionResults(BaseModel):
    predictions: List[float]

class TestSalesForecasting:
    def __init__(self):
        self.model_rf = None

    def load_random_forest_model(self, model_filename):
        """Loads the saved model from a file."""
        try:
            self.model_rf = joblib.load(model_filename)
            logging.info(f"Model loaded from {model_filename}.")
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {model_filename}")
            raise e

    def preprocess_test_data(self, test_data):
        """Preprocesses the test data to match the training data preprocessing steps."""
        preprocessor = ModelPrepro(test_data)
        preprocessor.data_prepro()
        preprocessor.days_to_next_holiday()
        preprocessor.days_after_last_holiday()
        preprocessor.feature_engineering()
        preprocessor.handel_missing()

        processed_test_data = preprocessor.data

        # Optionally drop the 'Date' column if it exists
        if 'Date' in processed_test_data.columns:
            processed_test_data.drop(columns=['Date'], inplace=True)

        # Ensure only numeric columns remain
        processed_test_data = processed_test_data.select_dtypes(include=[np.number])
        processed_test_data.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

        return processed_test_data

    def run_random_forest_predictions(self, test_data, model_file_path):
        """Runs predictions using the pre-trained Random Forest model on the test data."""
        if self.model_rf is None:
            self.load_random_forest_model(model_file_path)

        # Drop the 'Id' column from the test data before preprocessing, if it exists
        if 'Id' in test_data.columns:
            test_data = test_data.drop(columns=['Id'])

        processed_test_data = self.preprocess_test_data(test_data)

        # Predict using the preprocessed test data
        predictions = self.model_rf.predict(processed_test_data)

        return predictions  # Return only predictions (array of predicted sales values)

# Create an instance of the TestSalesForecasting class
sales_forecasting = TestSalesForecasting()

# Define the model file path
model_file_path = "/10 A KAI 2/Week 4/Sales_Forcasting/model/random_forest_model_2024-09-24-14-41-47.pkl"

@app.post("/predict-sales/", response_model=PredictionResults)
async def predict_sales(file: UploadFile = File(...)):
    try:
        # Read and decode the file contents
        contents = await file.read()
        test_data = pd.read_csv(StringIO(contents.decode('utf-8')))
        logging.info(f"Received test data with {len(test_data)} rows.")
        
        # Run predictions
        predictions = sales_forecasting.run_random_forest_predictions(test_data, model_file_path)

        # Return only the predictions (list of floats)
        response_data = {"predictions": list(predictions)}

        # Debugging: Print the response structure before returning
        logging.info(f"Response structure: {json.dumps(response_data, indent=2)}")

        # Return the response
        return response_data

    except FileNotFoundError as e:
        logging.error(f"Model file not found: {str(e)}")
        raise HTTPException(status_code=404, detail="Model file not found.")
    except pd.errors.EmptyDataError:
        logging.error("No data found in uploaded file.")
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except pd.errors.ParserError:
        logging.error("Error parsing the CSV file.")
        raise HTTPException(status_code=400, detail="Error parsing the CSV file.")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")
