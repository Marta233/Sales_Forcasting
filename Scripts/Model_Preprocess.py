import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
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
        # Convert Date to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Log the shape of the data
        logging.info(f'Starting data preprocessing for {self.data.shape[0]} rows and {self.data.shape[1]} columns.')
        
        # Log the descriptive statistics
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
        future_holidays = holiday_dates[holiday_dates > current_date]
        return (future_holidays[0] - current_date).days if len(future_holidays) > 0 else np.nan

    def _calculate_days_after_last_holiday(self, current_date, holiday_dates):
        past_holidays = holiday_dates[holiday_dates < current_date]
        return (current_date - past_holidays[-1]).days if len(past_holidays) > 0 else np.nan
    def feature_engineering(self):
        self.data['weekend'] = self.data['DayOfWeek'].apply(lambda x: 1 if x > 5 else 0)
        self.data['weekdays'] = self.data['DayOfWeek'].apply(lambda x: 1 if x <= 5 else 0)
        self.data['Quarter'] = self.data['Date'].dt.quarter
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Seasons'] = self.data['Month'].apply(lambda x: 1 if 3 <= x <= 6 else 2 if 7 <= x <= 9 else 3 if 10 <= x <= 12 else 4)
        self.data['Sales_lag_1'] = self.data['Sales'].shift(1)  # Sales on the previous day
        self.data['Sales_lag_7'] = self.data['Sales'].shift(7)  # Sales 7 days ago

    def handel_missing(self):
        # Calculate the percentage of missing values for each column
        missing_percentage = self.data.isnull().mean() * 100

        # Identify columns with more than 32% missing values
        columns_to_remove = missing_percentage[missing_percentage > 31].index

        # Remove those columns from the DataFrame
        self.data.drop(columns=columns_to_remove, inplace=True)
        # Optionally, print the remaining columns and their missing percentages
        remaining_missing_percentage = self.data.isnull().mean() * 100
        self.data['DaysToNextHoliday'].fillna(self.data['DaysToNextHoliday'].mean(), inplace=True)
        self.data['DaysAfterLastHoliday'].fillna(self.data['DaysAfterLastHoliday'].mean(), inplace=True)
        self.data['CompetitionDistance'].fillna(self.data['CompetitionDistance'].mean(), inplace=True)
        mean_sales_lag_1 = self.data['Sales_lag_1'].mean()
        mean_sales_lag_7 = self.data['Sales_lag_7'].mean()
        self.data['Sales_lag_1'].fillna(mean_sales_lag_1, inplace=True)
        self.data['Sales_lag_7'].fillna(mean_sales_lag_7, inplace=True)
        # Transforming the 'IsStateHoliday' column
        self.data['IsStateHoliday'] = self.data['StateHoliday'].apply(lambda x: 0 if x == 0 else 1)
        self.data['StateHoliday'].dropna(inplace=True)
        return remaining_missing_percentage
    def salse_othe_futur_corr(self):
        # Ensure the data has been encoded
        if not self.encoded:
            raise ValueError("Data must be encoded before performing other operations.")
        
        # Calculate the correlation matrix
        correlation_matrix = self.data.corr()

        # Display the correlation matrix as a table
        print("Correlation Matrix:")
        print(correlation_matrix)

        # Visualize the correlation matrix as a heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()
    def encode_categorical_features(self):
        categorical_cols = ['Assortment', 'StoreType']
        
        for col in categorical_cols:
            if col in self.data.columns:
                # Convert to string and fill missing values
                self.data[col] = self.data[col].astype(str).fillna('missing')
        
        # One-hot encoding for specified columns
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)
        self.encoded = True
        logging.info("One-hot encoding completed. Encoded columns: {}".format(self.data.columns.tolist()))
    def scale_data(self):
        self.data.drop(columns=['Date'], inplace=True)
        features_to_scale = ['CompetitionDistance','Customers', 'Sales_lag_1',	'Sales_lag_7','DaysToNextHoliday', 'DaysAfterLastHoliday']
        scaler = StandardScaler()
        self.data[features_to_scale] = scaler.fit_transform(self.data[features_to_scale])
        logging.info('Data scaling complete.')
    # Calculate correlation matrix
   
    
    def future_selection_corr(self):
        # Convert boolean columns to integers
        bool_cols = self.data.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            self.data[col] = self.data[col].astype(int)

        # Select only numeric columns for correlation
        numeric_data = self.data.select_dtypes(include=['number'])

        # Calculate the correlation matrix
        correlation_matrix = numeric_data.corr()

        # Display the correlation matrix as a table
        print("Correlation Matrix:")
        print(correlation_matrix)

        # Visualize the correlation matrix as a heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()
class SalesForecasting:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.preprocessed_data = None
        self.model_rf = None

    def load_data(self):
        """Loads data from a CSV file and stores it in the class instance."""
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
        """Applies preprocessing steps to the data including feature engineering and scaling."""
        if self.data is None:
            logging.error("Data not loaded. Please run load_data() first.")
            return

        # Assuming 'ModelPrepro' is a separate class responsible for preprocessing
        preprocessor = ModelPrepro(self.data)
        preprocessor.data_prepro()
        preprocessor.days_to_next_holiday()
        preprocessor.days_after_last_holiday()
        preprocessor.feature_engineering()
        preprocessor.handel_missing()
        preprocessor.encode_categorical_features()  # Ensure this is called before dropping columns
        preprocessor.scale_data()

        self.preprocessed_data = preprocessor.data

        # Optionally drop the Date column if still present
        if 'Date' in self.preprocessed_data.columns:
            self.preprocessed_data.drop(columns=['Date'], inplace=True)

        # Ensure only numeric columns remain after encoding
        self.preprocessed_data = self.preprocessed_data.select_dtypes(include=[np.number])
        self.preprocessed_data.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        logging.info(f'Preprocessed data shape: {self.preprocessed_data.shape}')
        logging.info(f'Available columns after preprocessing: {self.preprocessed_data.columns.tolist()}')
        return self.preprocessed_data

    def fit_random_forest_model(self):
        """Fits a Random Forest model using the preprocessed data."""
        if self.preprocessed_data is None:
            logging.error("Data not preprocessed. Please run preprocess_data() first.")
            return


      
        X = self.preprocessed_data.drop(columns=['Sales'])
        y = self.preprocessed_data['Sales']

        # Create a pipeline
        self.model_rf = Pipeline(steps=[
            ('scaler', StandardScaler()),  # Scaling step
            ('model_rf', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))  # Model
        ])

        # Fit the model
        self.model_rf.fit(X, y)
        logging.info('Random Forest model fitting complete.')

        # Feature importance
        feature_importances = self.model_rf.named_steps['model_rf'].feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title("Random Forest Feature Importances")
        plt.show()

    def evaluate_model(self):
        if self.preprocessed_data is None:
            logging.error("Data not preprocessed. Please run preprocess_data() first.")
            return

       
      
        X_rf = self.preprocessed_data.drop(columns=['Sales'] )
        y_rf = self.preprocessed_data['Sales']

        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

        rf_predictions = self.model_rf.predict(X_test_rf)

        rf_mae = mean_absolute_error(y_test_rf, rf_predictions)
        rf_r2 = r2_score(y_test_rf, rf_predictions)

        lower_bound, upper_bound = self.estimate_confidence_intervals(X_test_rf)

        logging.info(f'Random Forest - MAE: {rf_mae}, R^2: {rf_r2}')
        logging.info(f'Confidence Interval: [{lower_bound}, {upper_bound}]')

    def estimate_confidence_intervals(self, X_test, n_iterations=1000, alpha=0.05):
        """Estimate confidence intervals for predictions using bootstrap sampling."""
        predictions = []

        for _ in range(n_iterations):
            X_bootstrap, _, y_bootstrap, _ = train_test_split(self.preprocessed_data.drop(columns=['Sales']),
                                                            self.preprocessed_data['Sales'],
                                                            test_size=0.2, random_state=np.random.randint(0, 10000))
            model_rf_bootstrap = RandomForestRegressor(n_estimators=50, random_state=42)
            model_rf_bootstrap.fit(X_bootstrap, y_bootstrap)
            pred = model_rf_bootstrap.predict(X_test)
            predictions.append(pred)

        predictions = np.array(predictions)
        lower_bound = np.percentile(predictions, 100 * alpha / 2, axis=0)
        upper_bound = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)

        return lower_bound, upper_bound

    def save_random_forest_model(self):
        if self.model_rf is None:
            logging.error('Model not trained. Please fit the model first.')
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_name = f'random_forest_model_{timestamp}.pkl'
        joblib.dump(self.model_rf, file_name)
        logging.info(f'Random Forest model saved as {file_name}')
