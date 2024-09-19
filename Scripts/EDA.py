import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BASICEDA:
    def __init__(self, df_train, df_test, df_store):
        self.df_train = df_train
        self.df_test = df_test
        self.df_store = df_store
        self.holidays = None  # Initialize holidays as an instance variable
        self.school_holidays = None  # Initialize school_holidays as an instance variable

    def check_distribution_train_test_data(self):
        logging.info("Checking the distribution of promotions in the training and test datasets...")
        train_promo_counts = self.df_train['Promo'].value_counts(normalize=True)
        test_promo_counts = self.df_test['Promo'].value_counts(normalize=True) 
        print(train_promo_counts)
        print(test_promo_counts)

        # Combine counts into a DataFrame for easier plotting
        distribution_df = pd.DataFrame({
            'Train': train_promo_counts,
            'Test': test_promo_counts
        }).fillna(0)

        # Plotting distributions
        distribution_df.plot(kind='bar', figsize=(8, 5))
        plt.title('Promotion Distribution in Training vs Test Set')
        plt.ylabel('Proportion')
        plt.xlabel('Promotion (0 = No Promo, 1 = Promo)')
        plt.xticks(rotation=0)
        plt.legend(title='Dataset')
        plt.show()

    def merge_store_data(self):
        logging.info("Merging training data with store data...")
        self.df_train = self.df_train.merge(self.df_store, on='Store', how='left')
        logging.info("Merging completed. New training data shape: {}".format(self.df_train.shape))
        return self.df_train

    def df_basic_info(self):
        logging.info("Describing the training data...")
        return round(self.df_train.describe().T,2)

    def missing_percentage(self):
        # Calculate the percentage of missing values
        missing_percent = self.df_train.isnull().sum() / len(self.df_train) * 100
        
        # Create a DataFrame to display the results nicely
        missing_df = pd.DataFrame({
            'Column': self.df_train.columns,
            'Missing Percentage': missing_percent
        }).sort_values(by='Missing Percentage', ascending=False)
        
        return missing_df

    def handle_missing_values(self):
        logging.info("Handling missing values in the training data...")
        numeric_cols = self.df_train.select_dtypes(include=['float64', 'int64']).columns
        self.df_train[numeric_cols] = self.df_train[numeric_cols].fillna(self.df_train[numeric_cols].median())

        categorical_cols = self.df_train.select_dtypes(include=['object']).columns
        self.df_train[categorical_cols] = self.df_train[categorical_cols].fillna(self.df_train[categorical_cols].mode().iloc[0])

        logging.info("Missing values handled. New training data shape: {}".format(self.df_train.shape))
        return self.df_train

    def data_types(self):
        data_typs = self.df_train.dtypes
        return pd.DataFrame({
            'Column': self.df_train.columns,
            'Data Type': data_typs
        }).sort_values(by='Data Type', ascending=False)

    def outlier_check_perc(self):
        numeric_df = self.df_train.select_dtypes(include=[float, int])
        if numeric_df.empty:
            raise ValueError("No numeric columns available for outlier detection.")

        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
        total_values = numeric_df.count()
        outlier_percentage = (outliers.sum() / total_values) * 100
        
        return pd.DataFrame({
            'Column': numeric_df.columns,
            'Outlier Percentage': outlier_percentage
        }).sort_values(by='Outlier Percentage', ascending=False)

    def analyze_sales_holidays(self):
        logging.info("Analyzing sales data for holiday effects...")
        df = self.df_train.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['IsStateHoliday'] = df['StateHoliday'].isin(['a', 'b', 'c'])
        df['IsSchoolHoliday'] = df['SchoolHoliday'] == 1
        
        # Extract unique dates for state holidays
        self.holidays = df[df['IsStateHoliday']]['Date'].dt.date.unique()
        
        # Extract unique dates for school holidays
        self.school_holidays = df[df['IsSchoolHoliday']]['Date'].dt.date.unique()

        # Apply the categorization function
        df['Category'] = df.apply(lambda row: self.categorize_sales(row), axis=1)
        
        sales_summary = self.calculate_sales_metrics(df)
        self.visualize_total_sales(sales_summary)
        self.visualize_average_sales(sales_summary)
        return sales_summary

    def categorize_sales(self, row):
        if row['IsStateHoliday']:
            return 'During State Holiday'
        elif row['IsSchoolHoliday']:
            return 'During School Holiday'
        elif (row['Date'] - pd.DateOffset(days=1)).date() in self.holidays:
            return 'Before State Holiday'
        elif (row['Date'] + pd.DateOffset(days=1)).date() in self.holidays:
            return 'After State Holiday'
        elif (row['Date'] - pd.DateOffset(days=1)).date() in self.school_holidays:
            return 'Before School Holiday'
        elif (row['Date'] + pd.DateOffset(days=1)).date() in self.school_holidays:
            return 'After School Holiday'
        return 'Regular Day'


    def calculate_sales_metrics(self, data):
        return data.groupby('Category').agg(
            Total_Sales=('Sales', 'sum'),
            Average_Sales=('Sales', 'mean'),
            Total_Customers=('Customers', 'sum'),
            Average_Customers=('Customers', 'mean')
        ).reset_index()

    def visualize_total_sales(self, sales_summary):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Category', y='Total_Sales', data=sales_summary)
        plt.title('Total Sales by Category')
        plt.ylabel('Total Sales')
        plt.xlabel('Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def visualize_average_sales(self, sales_summary):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Category', y='Average_Sales', data=sales_summary)
        plt.title('Average Sales by Category')
        plt.ylabel('Average Sales')
        plt.xlabel('Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    def seasonal_purchase_treand_ana(self):
        logging.info("Analyzing seasonal purchase trends...")
        df = self.df_train.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        seasonal_summary = self.calculate_seasonal_sales_metrics(df)
        self.visualize_seasonal_sales(seasonal_summary)
        return seasonal_summary

    def calculate_seasonal_sales_metrics(self, data):
        return data.groupby('Month').agg(
            Total_Sales=('Sales', 'sum'),
            Average_Sales=('Sales', 'mean'),
            Total_Customers=('Customers', 'sum'),
            Average_Customers=('Customers', 'mean')
        ).reset_index()

    def visualize_seasonal_sales(self, seasonal_summary):
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Month', y='Total_Sales', data=seasonal_summary)
        plt.title('Total Sales by Month')
        plt.ylabel('Total Sales')
        plt.xlabel('Month')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.tight_layout()
        plt.show()
