import pandas as pd
import logging
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BASICEDA:
    def __init__(self, df_train, df_test, df_store):
        self.df_train = df_train
        self.df_test = df_test
        self.df_store = df_store
    def check_distribution_train_test_data(self):
        logging.info("Checking the distribution of promotions in the training and test datasets...")
        # Calculate the distribution of promotions
        train_promo_counts = self.df_train['Promo'].value_counts(normalize=True)
        test_promo_counts = self.df_test['Promo'].value_counts(normalize=True) 

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
        print(self.df_train.describe())
