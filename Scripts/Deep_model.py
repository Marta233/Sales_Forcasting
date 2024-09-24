import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelPrepro:
    def __init__(self, data):
        self.data = data

    def data_prepro(self):
        """Initial preprocessing steps."""
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        logging.info(f'Starting data preprocessing for {self.data.shape[0]} rows and {self.data.shape[1]} columns.')
        description = self.data.describe().T
        logging.info('Descriptive statistics of the dataset:')
        logging.info(description)

    def days_to_next_holiday(self):
        holiday_dates = self.data[self.data['StateHoliday'] != '0']['Date'].sort_values().unique()
        self.data['DaysToNextHoliday'] = self.data['Date'].apply(lambda x: self._calculate_days_to_next_holiday(x, holiday_dates))

    def days_after_last_holiday(self):
        holiday_dates = self.data[self.data['StateHoliday'] != '0']['Date'].sort_values().unique()
        self.data['DaysAfterLastHoliday'] = self.data['Date'].apply(lambda x: self._calculate_days_after_last_holiday(x, holiday_dates))

    def _calculate_days_to_next_holiday(self, current_date, holiday_dates):
        # Ensure current_date is a datetime object
        current_date = pd.to_datetime(current_date)
        future_holidays = holiday_dates[holiday_dates > current_date]
        return (future_holidays[0] - current_date).days if len(future_holidays) > 0 else np.nan

    def _calculate_days_after_last_holiday(self, current_date, holiday_dates):
        # Ensure current_date is a datetime object
        current_date = pd.to_datetime(current_date)
        past_holidays = holiday_dates[holiday_dates < current_date]
        return (current_date - past_holidays[-1]).days if len(past_holidays) > 0 else np.nan
    def feature_engineering(self):
        self.data['weekend'] = self.data['DayOfWeek'].apply(lambda x: 1 if x > 5 else 0)
        self.data['weekdays'] = self.data['DayOfWeek'].apply(lambda x: 1 if x <= 5 else 0)
        self.data['Quarter'] = self.data['Date'].dt.quarter
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Seasons'] = self.data['Month'].apply(lambda x: 1 if 3 <= x <= 6 else 2 if 7 <= x <= 9 else 3 if 10 <= x <= 12 else 4)
        self.data['IsStateHoliday'] = self.data['StateHoliday'].apply(lambda x: 0 if x == 0 else 1)
        self.data['StateHoliday'].dropna(inplace=True)

    def handle_missing(self):
        self.data['DaysToNextHoliday'].fillna(self.data['DaysToNextHoliday'].mean(), inplace=True)
        self.data['DaysAfterLastHoliday'].fillna(self.data['DaysAfterLastHoliday'].mean(), inplace=True)

    def encode_categorical_features(self):
        categorical_cols = ['Assortment', 'StoreType']
        for col in categorical_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(str).fillna('missing')
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)
        logging.info("One-hot encoding completed. Encoded columns: {}".format(self.data.columns.tolist()))

    def scale_data(self):
        self.data.drop(columns=['Date'], inplace=True)
        features_to_scale = [ 'DaysToNextHoliday', 'DaysAfterLastHoliday']
        scaler = StandardScaler()
        self.data[features_to_scale] = scaler.fit_transform(self.data[features_to_scale])
        logging.info('Data scaling complete.')

class SalesForecasting:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.preprocessed_data = None
        self.model_lstm = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info(f'Data loaded with shape: {self.data.shape}')
        except FileNotFoundError as e:
            logging.error(f"File not found: {self.file_path}")
            raise e
        except Exception as e:
            logging.error(f"An error occurred while loading data: {e}")
            raise e

    def preprocess_data(self):
        if self.data is None:
            logging.error("Data not loaded. Please run load_data() first.")
            return

        preprocessor = ModelPrepro(self.data)
        preprocessor.data_prepro()  # Ensure this is called to convert Date to datetime
        preprocessor.days_to_next_holiday()
        preprocessor.days_after_last_holiday()
        preprocessor.feature_engineering()
        preprocessor.handle_missing()
        # preprocessor.encode_categorical_features()
        preprocessor.scale_data()

        self.preprocessed_data = preprocessor.data

        if 'Date' in self.preprocessed_data.columns:
            self.preprocessed_data.drop(columns=['Date'], inplace=True)

        self.preprocessed_data = self.preprocessed_data.select_dtypes(include=[np.number])
        logging.info(f'Preprocessed data shape: {self.preprocessed_data.shape}')
        logging.info(f'Available columns after preprocessing: {self.preprocessed_data.columns.tolist()}')
        return self.preprocessed_data

    def create_lstm_dataset(self, time_step=10):
        """Transforms the data into a format suitable for LSTM."""
        features = self.preprocessed_data.drop(columns=['Sales']).values
        target = self.preprocessed_data['Sales'].values

        X, y = [], []
        for i in range(len(features) - time_step):
            X.append(features[i:i + time_step])
            y.append(target[i + time_step])
        
        return np.array(X), np.array(y)

    def fit_lstm_model(self, time_step=10, epochs=20, batch_size=32):
        """Fits an LSTM model using the preprocessed data."""
        if self.preprocessed_data is None:
            logging.error("Data not preprocessed. Please run preprocess_data() first.")
            return

        X, y = self.create_lstm_dataset(time_step)
        self.model_lstm = Sequential()
        self.model_lstm.add(LSTM(20, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        self.model_lstm.add(Dropout(0.2))
        self.model_lstm.add(LSTM(20))
        self.model_lstm.add(Dropout(0.2))
        self.model_lstm.add(Dense(1))
        
        self.model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        self.model_lstm.fit(X, y, epochs=epochs, batch_size=batch_size)
        logging.info('LSTM model fitting complete.')

    def evaluate_lstm_model(self, time_step=10):
        """Evaluates the LSTM model on a test set."""
        if self.preprocessed_data is None:
            logging.error("Data not preprocessed. Please run preprocess_data() first.")
            return

        X, y = self.create_lstm_dataset(time_step)
        y_pred = self.model_lstm.predict(X)

        # Calculate metrics: MAE and R^2 score
        mae = mean_absolute_error(y[time_step:], y_pred)
        r2 = r2_score(y[time_step:], y_pred)

        logging.info(f'LSTM Model - MAE: {mae}, R^2: {r2}')

    def save_lstm_model(self):
        """Saves the trained LSTM model to a file with a timestamp."""
        if self.model_lstm is None:
            logging.error('Model not trained. Please fit the model first.')
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_name = f'lstm_model_{timestamp}.h5'
        self.model_lstm.save(file_name)
        
        logging.info(f'LSTM model saved as {file_name}')

