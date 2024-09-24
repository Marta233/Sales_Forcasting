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
    Predicted_Sales: float

class TestSalesForecasting:
    def __init__(self):
        self.model_rf = None

    def load_random_forest_model(self, model_filename):
        try:
            self.model_rf = joblib.load(model_filename)
            logging.info(f"Model loaded from {model_filename}.")
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {model_filename}")
            raise e
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise e

    def preprocess_test_data(self, test_data):
        preprocessor = ModelPrepro(test_data)
        preprocessor.data_prepro()
        preprocessor.days_to_next_holiday()
        preprocessor.days_after_last_holiday()
        preprocessor.feature_engineering()
        preprocessor.handel_missing()

        processed_test_data = preprocessor.data

        # Drop unnecessary columns
        if 'Date' in processed_test_data.columns:
            processed_test_data.drop(columns=['Date'], inplace=True)

        processed_test_data = processed_test_data.select_dtypes(include=[np.number])
        processed_test_data.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        processed_test_data.drop(columns=['Id'], inplace=True, errors='ignore')

        return processed_test_data

    def run_random_forest_predictions(self, test_data):
        if self.model_rf is None:
            logging.error("Model not loaded.")
            raise Exception("Model not loaded.")

        processed_test_data = self.preprocess_test_data(test_data)

        predictions = self.model_rf.predict(processed_test_data)

        return predictions.tolist()  # Return predictions as a list

sales_forecasting = TestSalesForecasting()

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

        sales_forecasting.load_random_forest_model(model_file_path)

        predictions = sales_forecasting.run_random_forest_predictions(test_data)

        # Convert the predictions list to a list of SalesPredictionResponse
        results = [SalesPredictionResponse(Predicted_Sales=pred) for pred in predictions]
        return results

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