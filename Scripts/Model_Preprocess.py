import pandas as pd
import numpy as np
import logging
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
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
    def __init__(self, df_train, df_store):
        self.df_train = df_train
        self.df_store = df_store
    
    def merge_store_data(self):
        logging.info("Merging training df_train with store df_train...")
        self.df_train = self.df_train.merge(self.df_store, on='Store', how='left')
        logging.info("Merging completed. New training df_train shape: {}".format(self.df_train.shape))
        return self.df_train
    def data_prepro(self):
        # Convert Date to datetime
        self.df_train['Date'] = pd.to_datetime(self.df_train['Date'])
        
        # Log the shape of the df_train
        logging.info(f'Starting df_train preprocessing for {self.df_train.shape[0]} rows and {self.df_train.shape[1]} columns.')
        
        # Log the descriptive statistics
        logging.info('Descriptive statistics of the df_trainset:')
        description = self.df_train.describe().T.round(2)
        return description
    def days_to_next_holiday(self):
        holiday_dates = self.df_train[self.df_train['StateHoliday'] != '0']['Date'].sort_values().unique()
        self.df_train['DaysToNextHoliday'] = self.df_train['Date'].apply(lambda x: self._calculate_days_to_next_holiday(x, holiday_dates))

    def days_after_last_holiday(self):
        holiday_dates = self.df_train[self.df_train['StateHoliday'] != '0']['Date'].sort_values().unique()
        self.df_train['DaysAfterLastHoliday'] = self.df_train['Date'].apply(lambda x: self._calculate_days_after_last_holiday(x, holiday_dates))

    def _calculate_days_to_next_holiday(self, current_date, holiday_dates):
        future_holidays = holiday_dates[holiday_dates > current_date]
        return (future_holidays[0] - current_date).days if len(future_holidays) > 0 else np.nan

    def _calculate_days_after_last_holiday(self, current_date, holiday_dates):
        past_holidays = holiday_dates[holiday_dates < current_date]
        return (current_date - past_holidays[-1]).days if len(past_holidays) > 0 else np.nan
    def feature_engineering(self):
        self.df_train['weekend'] = self.df_train['DayOfWeek'].apply(lambda x: 1 if x > 5 else 0)
        self.df_train['weekdays'] = self.df_train['DayOfWeek'].apply(lambda x: 1 if x <= 5 else 0)
        self.df_train['Quarter'] = self.df_train['Date'].dt.quarter
        self.df_train['Month'] = self.df_train['Date'].dt.month
        # self.df_train['Seasons'] = self.df_train['Month'].apply(lambda x: 1 if 3 <= x <= 6 else 2 if 7 <= x <= 9 else 3 if 10 <= x <= 12 else 4)
        self.df_train['Sales_lag_1'] = self.df_train['Sales'].shift(1)  # Sales on the previous day
        self.df_train['Sales_lag_7'] = self.df_train['Sales'].shift(7)  # Sales 7 days ago
        return self.df_train
    def missing_percentage(self):
        # Calculate the percentage of missing values for each column
        missing_percentage = self.df_train.isnull().mean() * 100
        return missing_percentage
    def handel_missing(self):
        # Calculate the percentage of missing values for each column
        missing_percentage = self.df_train.isnull().mean() * 100
        columns_to_remove = missing_percentage[missing_percentage > 31].index
        self.df_train.drop(columns=columns_to_remove, inplace=True)
        self.df_train.drop(columns=['Date'], inplace=True)
        self.df_train['DaysToNextHoliday'].fillna(self.df_train['DaysToNextHoliday'].mean(), inplace=True)
        self.df_train['DaysAfterLastHoliday'].fillna(self.df_train['DaysAfterLastHoliday'].mean(), inplace=True)
        self.df_train['CompetitionDistance'].fillna(self.df_train['CompetitionDistance'].mean(), inplace=True)
        mean_sales_lag_1 = self.df_train['Sales_lag_1'].mean()
        mean_sales_lag_7 = self.df_train['Sales_lag_7'].mean()
        self.df_train['Sales_lag_1'].fillna(mean_sales_lag_1, inplace=True)
        self.df_train['Sales_lag_7'].fillna(mean_sales_lag_7, inplace=True)
        self.df_train['IsStateHoliday'] = self.df_train['StateHoliday'].apply(lambda x: 0 if x == 0 else 1)
        # self.df_train['StateHoliday'].dropna(inplace=True)
        self.df_train.drop(columns=['StateHoliday'], inplace=True)
        object_cols = self.df_train.select_dtypes(include=['object']).columns.tolist()
        # One-hot encode the object columns
        self.df_train = pd.get_dummies(self.df_train, columns=object_cols, drop_first=True)
        object_cols = self.df_train.select_dtypes(include=['bool']).columns.tolist()
        self.df_train[object_cols] = self.df_train[object_cols].astype(int)
        return self.df_train
    def salse_othe_futur_corr(self):
        correlation_matrix = self.df_train.corr()
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()
class SalesForecasting:
    def __init__(self, file_path1,file_path2):
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.df_train = None
        self.df_store  = None
        self.preprocessed_data = None
        self.model_rf = None
        self.model_lstm = None
        self.scaler = None
        self.X = None
        self.y = None
    def load_data(self):
        """Loads df_train from a CSV file and stores it in the class instance."""
        try:
            self.df_train = pd.read_csv(self.file_path1)
            self.df_store = pd.read_csv(self.file_path2)
            logging.info(f'df_train loaded with shape: {self.df_train.shape}')
        except FileNotFoundError as e:
            logging.error(f"File not found: {self.file_path1}")
            raise e
        except Exception as e:
            logging.error(f"An error occurred while loading df_train: {e}")
            raise e

    def preprocess_data(self):
        """Applies preprocessing steps to the df_train including feature engineering and scaling."""
        preprocessor = ModelPrepro(self.df_train, self.df_store)
        preprocessor.merge_store_data()
        preprocessor.data_prepro()
        preprocessor.days_to_next_holiday()
        preprocessor.days_after_last_holiday()
        preprocessor.feature_engineering()
        df = preprocessor.handel_missing()
        self.preprocessed_data = df
        logging.info(f'Preprocessed df_train shape: {self.preprocessed_data.shape}')
        logging.info(f'Available columns after preprocessing: {self.preprocessed_data.columns.tolist()}')

    def evaluate_model(self,x,y,model):
        y_pred = model.predict(x)
        r2 = r2_score(y_pred,y)
        MAE = mean_absolute_error(y_pred,y)
        MSE = mean_squared_error(y_pred,y)
        print('R^2 Score:', r2)
        print('Mean Absolute Error (MAE):', MAE)
        print('Mean Squared Error (MSE):', MSE)
    def fit_random_forest_model(self):
        # Split the data
        X = self.preprocessed_data.drop(columns=['Sales'])
        y = self.preprocessed_data['Sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=42))
        ])

        # Fit pipeline
        pipeline.fit(X_train, y_train)
        logging.info('Random Forest model fitting complete.')

        # Save the model
        self.model_rf = pipeline
        joblib.dump(self.model_rf, 'random_forest_model.pkl')

        # Save feature names
        feature_names = X.columns.tolist()
        with open('feature_names.pkl', 'wb') as f:
            joblib.dump(feature_names, f)

        # Feature importance
        feature_importances = self.model_rf.named_steps['rf'].feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title("Random Forest Feature Importances")
        plt.show()
        self.evaluate_model(X_test,  y_test, self.model_rf)
    from sklearn.preprocessing import StandardScaler

    def fit_lstm_model_simple(self, n_steps=10):
        print("Scaling and training LSTM model...")

        # 1. Scale the whole preprocessed_data (including 'Sales')
        self.scaler_all = StandardScaler()
        scaled_data = self.scaler_all.fit_transform(self.preprocessed_data)
        scaled_df = pd.DataFrame(scaled_data, columns=self.preprocessed_data.columns)

        # 2. Now split features and target
        X = scaled_df.drop(columns=['Sales'])
        y = scaled_df[['Sales']]  # Keep y as DataFrame

        # 3. Create sequences
        X_seq, y_seq = [], []
        for i in range(n_steps, len(X)):
            X_seq.append(X.iloc[i - n_steps:i].values)
            y_seq.append(y.iloc[i].values)
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # 4. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

        # 5. Build and compile model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # 6. Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # 7. Save model and scaler
        model.save('lstm_model.h5')
        self.model_lstm = model
        self.evaluate_model(X_test, y_test, model)

        # 8. Plot training loss
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.title("LSTM Training Loss")
        plt.show()

        print("LSTM training done!")
