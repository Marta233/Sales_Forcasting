import pandas as pd
import logging
import joblib
import sys
import numpy as np

sys.path.insert(0, '/10 A KAI 2/Week 4/Sales_Forcasting/')
from Scripts.Model_Preprocess import SalesForecasting
from Scripts.Model_build import ModelPrepro


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
        # Assuming 'ModelPrepro' is a separate class responsible for preprocessing
        preprocessor = ModelPrepro(test_data)
        preprocessor.data_prepro()
        preprocessor.days_to_next_holiday()
        preprocessor.days_after_last_holiday()
        preprocessor.feature_engineering()
        # preprocessor.feature_engineering1
        preprocessor.handel_missing()
        # preprocessor.encode_categorical_features()

        processed_test_data = preprocessor.data

        # Optionally drop the 'Date' column if it exists
        if 'Date' in processed_test_data.columns:
            processed_test_data.drop(columns=['Date'], inplace=True)

        # Ensure only numeric columns remain
        processed_test_data = processed_test_data.select_dtypes(include=[np.number])
        processed_test_data.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        processed_test_data.drop(columns=['Id'], inplace=True, errors='ignore')

        return processed_test_data

    def run_random_forest_predictions(self, test_data, model_file_path):
        """Runs predictions using the pre-trained Random Forest model on the test data."""
        # Load the model if not already loaded
        if self.model_rf is None:
            self.load_random_forest_model(model_file_path)

        # Preprocess the test data
        processed_test_data = self.preprocess_test_data(test_data)

        # Keep the 'Id' column from the original test data
        if 'Id' in test_data.columns:
            id_column = test_data['Id']
        else:
            logging.warning("No 'Id' column found in test data. Assigning NaN to Id column.")
            id_column = pd.Series([np.nan] * len(test_data), name='Id')

        # Predict using the preprocessed test data
        predictions = self.model_rf.predict(processed_test_data)

        # Create a DataFrame to display Id and Predicted Sales together
        results_df = pd.DataFrame({
            'Id': id_column,
            'Predicted_Sales': predictions
        })

        return results_df
