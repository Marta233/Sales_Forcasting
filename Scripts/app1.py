import pandas as pd
import logging
import joblib
import sys
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import your custom modules
sys.path.insert(0, '/10 A KAI 2/Week 4/Sales_Forcasting/')
from Scripts.Model_Preprocess import SalesForecasting
from Scripts.Model_build import ModelPrepro

# Initialize FastAPI app
app = FastAPI()

class SalesPredictionResponse(BaseModel):
    Id: int  # Added Id field
    Predicted_Sales: float

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

        # Keep the 'Id' column from the original test data
        id_column = test_data['Id'] if 'Id' in test_data.columns else pd.Series([np.nan] * len(test_data))

        processed_test_data = self.preprocess_test_data(test_data)

        # Predict using the preprocessed test data
        predictions = self.model_rf.predict(processed_test_data)

        # Combine Ids and predictions into a list of tuples
        results = [(id_val, pred) for id_val, pred in zip(id_column, predictions)]
        return results  # Return list of tuples (Id, Predicted_Sales)

# Create an instance of the TestSalesForecasting class
sales_forecasting = TestSalesForecasting()

# Define the model file path
model_file_path = "/10 A KAI 2/Week 4/Sales_Forcasting/notebook/random_forest_model_2024-09-24-14-41-47.pkl"

@app.post("/predict-sales/", response_model=list[SalesPredictionResponse])
async def predict_sales(file: UploadFile = File(...)):
    """
    API endpoint to predict sales from an uploaded CSV file.
    The CSV should contain test data for prediction.
    """
    try:
        contents = await file.read()
        test_data = pd.read_csv(StringIO(contents.decode('utf-8')))
        logging.info(f"Received test data with {len(test_data)} rows.")

        # Run predictions
        results = sales_forecasting.run_random_forest_predictions(test_data, model_file_path)

        # Convert results to a list of SalesPredictionResponse
        response = [SalesPredictionResponse(Id=id_val, Predicted_Sales=pred) for id_val, pred in results]
        return response

    except FileNotFoundError as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=404, detail="Model file not found.")
    except pd.errors.EmptyDataError:
        logging.error("Error during prediction: No data found in uploaded file.")
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

# Run the app with `uvicorn filename:app --reload`