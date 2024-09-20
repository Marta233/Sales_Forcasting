import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from scipy import stats
import matplotlib.dates as mdates
# import statsmodels.api as sm
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
    def corr_customer_sales(self):
        logging.info("Analyzing customer sales correlations...")
        df = self.df_train.copy()
        
        # Ensure 'Customers' and 'Sales' are numeric and handle any non-numeric data
        df = df[['Customers', 'Sales']].apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with NaN values resulting from coercion
        df = df.dropna()
        
        # Compute the correlation matrix
        corr_matrix = df.corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(8,6))
        sns.lineplot(x='Customers', y='Sales', data=df)
        plt.title('Customer Sales Relationship')
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.tight_layout()
        plt.show()
    
    def analyze_promotions(self):
    # Check if the necessary columns exist
        df = self.df_train.copy()
        required_columns = ['Promo', 'Sales', 'Customers']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate average sales by promotion status
        promo_sales = df.groupby('Promo')['Sales'].mean().reset_index()
        promo_customers = df.groupby('Promo')['Customers'].mean().reset_index()

        # Print average sales and customers by promotion status
        print("Average Sales by Promotion Status:")
        print(promo_sales)
        print("\nAverage Customers by Promotion Status:")
        print(promo_customers)

        # Visualization of average sales
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Promo', y='Sales', data=promo_sales)
        plt.title('Average Sales by Promotion Status')
        plt.ylabel('Average Sales')
        plt.xlabel('Promotion Active (0 = No, 1 = Yes)')
        plt.xticks(rotation=0)
        plt.show()

        # Visualization of average customers
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Promo', y='Customers', data=promo_customers)
        plt.title('Average Customers by Promotion Status')
        plt.ylabel('Average Customers')
        plt.xlabel('Promotion Active (0 = No, 1 = Yes)')
        plt.xticks(rotation=0)
        plt.show()
            # T-test for sales during promotions vs. non-promotions
        sales_with_promo = df[df['Promo'] == 1]['Sales']
        sales_without_promo = df[df['Promo'] == 0]['Sales']
        t_stat, p_value = stats.ttest_ind(sales_with_promo, sales_without_promo)
        print(f'T-test for Sales: t-statistic = {t_stat}, p-value = {p_value}')

        # T-test for customer counts during promotions vs. non-promotions
        customers_with_promo = df[df['Promo'] == 1]['Customers']
        customers_without_promo = df[df['Promo'] == 0]['Customers']
        t_stat_customers, p_value_customers = stats.ttest_ind(customers_with_promo, customers_without_promo)
        print(f'T-test for Customers: t-statistic = {t_stat_customers}, p-value = {p_value_customers}')

        
    def analyze_and_plot_promotions(self, top_n=10):
        # Convert 'Date' to datetime format
        data = self.df_train.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Group by store and calculate average sales when promo is active vs inactive
        promo_analysis = data.groupby('Store').agg(
            avg_sales_promo=('Sales', lambda x: x[data['Promo'] == 1].mean()),
            avg_sales_no_promo=('Sales', lambda x: x[data['Promo'] == 0].mean()),
            total_customers=('Customers', 'sum'),
            open_days=('Open', 'sum')
        ).reset_index()
        
        # Calculate the effectiveness of the promo
        promo_analysis['promo_effectiveness'] = promo_analysis['avg_sales_promo'] - promo_analysis['avg_sales_no_promo']
        
        # Identify stores where promo effectiveness is positive
        effective_stores = promo_analysis[promo_analysis['promo_effectiveness'] > 0]
        
        # Sort by promo effectiveness and total customers
        effective_stores = effective_stores.sort_values(by=['promo_effectiveness', 'total_customers'], ascending=False)
        
        # Select top N stores based on total sales or effectiveness
        top_stores = effective_stores.nlargest(top_n, 'total_customers')['Store'].values
        
        # Melt the DataFrame for easier plotting with only top stores
        melted_data = effective_stores[effective_stores['Store'].isin(top_stores)].melt(
            id_vars='Store', 
            value_vars=['avg_sales_promo', 'avg_sales_no_promo'], 
            var_name='Promo_Status', 
            value_name='Average_Sales'
        )

        # Create the bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Store', y='Average_Sales', hue='Promo_Status', data=melted_data)
        
        plt.title('Average Sales with and without Promotions for Selected Stores')
        plt.xlabel('Store')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=45)
        plt.legend(title='Promotion Status')
        plt.tight_layout()
        plt.show()
    def analyze_customer_behavior(self):
        data = self.df_train.copy()

        # Convert 'Date' to datetime format
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Create new columns for hour and day
        data['Hour'] = data['Date'].dt.hour
        data['Day'] = data['Date'].dt.date

        # Filter for open stores
        open_stores = data[data['Open'] == 1]
        
        # Group by date and hour to calculate total customers during open hours
        hourly_customers_open = open_stores.groupby(['Day', 'Hour'])['Customers'].sum().reset_index()

        # Create a new datetime column for open hours
        hourly_customers_open['Datetime'] = pd.to_datetime(hourly_customers_open['Day']) + pd.to_timedelta(hourly_customers_open['Hour'], unit='h')

        # Filter for closed stores
        closed_stores = data[data['Open'] == 0]

        # Group by date and hour to calculate total customers during closed hours
        hourly_customers_close = closed_stores.groupby(['Day', 'Hour'])['Customers'].sum().reset_index()

        # Create a new datetime column for closed hours
        hourly_customers_close['Datetime'] = pd.to_datetime(hourly_customers_close['Day']) + pd.to_timedelta(hourly_customers_close['Hour'], unit='h')

        # Create the plot
        plt.figure(figsize=(14, 7))

        # Plot for open hours
        sns.lineplot(data=hourly_customers_open, x='Datetime', y='Customers', marker='o', label='Open Hours', color='blue')

        # Plot for closed hours
        sns.lineplot(data=hourly_customers_close, x='Datetime', y='Customers', marker='o', label='Closed Hours', color='orange')

        # Enhancing the plot
        plt.title('Customer Trends Over Time (Open vs Closed Hours)')
        plt.xlabel('Date and Time')
        plt.ylabel('Total Customers')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    def analyze_weekday_weekend_sales(self):
        data = self.df_train.copy()
        """
        Analyzes which stores are open on weekends and weekdays, and compares their sales.
        Produces a plot to visualize the results.
        
        Parameters:
        - data: DataFrame containing store data with 'DayOfWeek' and 'Open' columns.

        Returns:
        - Tuple of (stores open on both weekdays and weekends, stores closed on weekdays but open on weekends)
        """
        # Filter stores that are open on weekends (Saturday and Sunday)
        weekend_stores = data[(data['DayOfWeek'] >= 6) & (data['Open'] == 1)]
        weekend_open_stores = weekend_stores['Store'].unique()

        # Filter stores that are open on all weekdays (Monday to Friday)
        weekday_stores = data[(data['DayOfWeek'] >= 1) & (data['DayOfWeek'] <= 5) & (data['Open'] == 1)]
        weekday_open_stores = weekday_stores['Store'].unique()

        # Filter stores that are closed on weekdays
        weekday_stores_closed = data[(data['DayOfWeek'] >= 1) & (data['DayOfWeek'] <= 5) & (data['Open'] == 0)]
        weekday_closed_stores = weekday_stores_closed['Store'].unique()

        # Identify which weekend stores are also open on weekdays
        weekend_sales_open = weekend_stores[weekend_stores['Store'].isin(weekday_open_stores)]

        # Identify which weekend stores are closed on weekdays
        weekend_sales_closed = weekend_stores[weekend_stores['Store'].isin(weekday_closed_stores)]

        # Plot the results
        store_counts = {
            'Stores Open on Weekdays and Weekends': len(weekend_sales_open['Store'].unique()),
            'Stores Closed on Weekdays but Open on Weekends': len(weekend_sales_closed['Store'].unique())
        }
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(store_counts.keys()), y=list(store_counts.values()), palette="coolwarm")
        plt.title('Comparison of Stores Open on Weekdays vs Weekends')
        plt.ylabel('Number of Stores')
        plt.xlabel('Store Type')
        plt.show()
         # Create a combined DataFrame for sales plotting
        sales_data = pd.DataFrame({
            'DayOfWeek': sales_open.index,
            'Sales Open': sales_open.values,
            'Sales Closed': sales_closed.values
        }).fillna(0)  # Fill NaN values for days where stores may not be open

        # Plot the sales comparison
        plt.figure(figsize=(12, 10))
        sns.lineplot(data=sales_data, x='DayOfWeek', y='Sales Open', label='Stores Open Weekdays & Weekends', marker='o')
        sns.lineplot(data=sales_data, x='DayOfWeek', y='Sales Closed', label='Stores Closed Weekdays but Open Weekends', marker='o')
        
        plt.title('Sales Comparison Between Store Types Across the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Total Sales')
        plt.xticks([1, 2, 3, 4, 5, 6, 7], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.legend()
        plt.grid(True)
        plt.show()

    
        return weekend_sales_open['Store'].unique(), weekend_sales_closed['Store'].unique()
       


    def analyze_assortment_type_effect(self):
        # Use self.df_train directly
        data = self.df_train.copy()
        
        # Ensure 'Assortment' and 'Sales' columns exist
        if 'Assortment' not in data.columns or 'Sales' not in data.columns:
            raise KeyError("Required columns 'Assortment' or 'Sales' are missing from the DataFrame")
        
        # Group by assortment type and calculate total sales
        sales_by_assortment = data.groupby('Assortment')['Sales'].sum().reset_index()
        
        # Print sales summary for debugging
        print(sales_by_assortment)

        # Create a bar plot to visualize sales by assortment type
        plt.figure(figsize=(8, 4))
        sns.barplot(data=sales_by_assortment, x='Assortment', y='Sales', palette='viridis')
        
        plt.title('Total Sales by Assortment Type')
        plt.xlabel('Assortment Type')
        plt.ylabel('Total Sales')
        # plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
   
    # def line_plot_compiteterdistance_sales(self):
    def analyze_competition_impact(self):
        # Calculate correlation between competition distance and sales
        df = self.df_train.copy()
        correlation = df['Sales'].corr(df['CompetitionDistance'])
        print(f"Correlation between Sales and Competition Distance: {correlation:.2f}")
        
        # Plotting Sales vs. Competition Distance
        plt.figure(figsize=(8, 5))
        plt.scatter(df['CompetitionDistance'], df['Sales'], color='blue')
        plt.title('Sales vs. Competition Distance')
        plt.xlabel('Competition Distance (m)')
        plt.ylabel('Sales')
        plt.grid()
        plt.show()


    def analyze_competitor_effects(self):
        df = self.df_train.copy()
        # Step 1: Identify stores with NA competitor distance
        initial_na_competitors = df[df['CompetitionDistance'].isna()]
        
        # Step 2: Identify stores that later have valid competitor distance
        stores_with_na = initial_na_competitors['Store'].unique()
        stores_with_competitor_distance = df[df['Store'].isin(stores_with_na) & df['CompetitionDistance'].notna()]

        # Step 3: Find the relevant sales data for these stores
        combined_data = pd.merge(initial_na_competitors, stores_with_competitor_distance, on=['Store', 'Date'], suffixes=('_initial', '_after'))
        
        # Step 4: Prepare data for plotting
        sales_comparison = combined_data[['Store', 'Sales_initial', 'Sales_after', 'CompetitionDistance_after']]
        
        # Step 5: Plotting the sales comparison
        plt.figure(figsize=(12, 6))
        
        # Create a bar plot for sales comparison
        sales_comparison.set_index('Store')[['Sales_initial', 'Sales_after']].plot(kind='bar', color=['blue', 'orange'])
        
        plt.title('Sales Comparison Before and After Competitor Distance Recorded')
        plt.xlabel('Store')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.legend(['Sales Before Competitor Distance', 'Sales After Competitor Distance'])
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

        return sales_comparison


