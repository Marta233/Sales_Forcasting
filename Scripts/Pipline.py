import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from Scripts.Model_Preprocess import ModelPrepro  # Adjust the path as necessary

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SalesForecasting:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.preprocessed_data = None
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        logging.info(f'Data loaded with shape: {self.data.shape}')

    def preprocess_data(self):
        preprocessor = ModelPrepro(self.data)
        preprocessor.data_prepro()
        preprocessor.days_to_next_holiday()
        preprocessor.days_after_last_holiday()
        preprocessor.feature_engineering()
        preprocessor.handel_missing()
        preprocessor.scale_data()
        self.preprocessed_data = preprocessor.data

    def fit_model(self):
        X = self.preprocessed_data.drop(columns=['Sales'])  # Features
        y = self.preprocessed_data['Sales']  # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and fit the model
        self.model = RandomForestRegressor(random_state=42)
        self.model.fit(X_train, y_train)

        # Save test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        logging.info('Model fitting complete.')

    def make_predictions(self):
        return self.model.predict(self.X_test)

    def evaluate_model(self):
        predictions = self.make_predictions()
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        logging.info(f'Model evaluation - MSE: {mse}, R^2: {r2}')
        return mse, r2