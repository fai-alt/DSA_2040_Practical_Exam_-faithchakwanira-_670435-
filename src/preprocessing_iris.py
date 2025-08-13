"""
DSA 2040 Practical Exam - Task 1: Data Preprocessing and Exploration
Data Mining Section - Iris Dataset Preprocessing

This script implements data preprocessing for the Iris dataset:
1. Load/generate Iris dataset
2. Handle missing values
3. Normalize features
4. Explore data with visualizations
5. Split into train/test sets

Author: [Your Name]
Date: August 13, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing_iris.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IrisPreprocessor:
    def __init__(self, use_synthetic=False):
        """
        Initialize the Iris preprocessor
        Args:
            use_synthetic: If True, generate synthetic data instead of loading real Iris
        """
        self.use_synthetic = use_synthetic
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.target = None
        self.scaler = MinMaxScaler()
        
        # Create output directory if it doesn't exist
        os.makedirs('../images', exist_ok=True)
        os.makedirs('../data', exist_ok=True)
        
    def generate_synthetic_iris(self, n_samples=150):
        """
        Generate synthetic Iris-like data with 3 clusters
        Creates realistic patterns similar to the real Iris dataset
        """
        logger.info(f"Generating {n_samples} synthetic Iris-like samples")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define cluster centers (similar to real Iris)
        cluster_centers = {
            'setosa': {
                'sepal_length': 5.0,
                'sepal_width': 3.5,
                'petal_length': 1.5,
                'petal_width': 0.25
            },
            'versicolor': {
                'sepal_length': 6.0,
                'sepal_width': 2.8,
                'petal_length': 4.2,
                'petal_width': 1.3
            },
            'virginica': {
                'sepal_length': 6.5,
                'sepal_width': 3.0,
                'petal_length': 5.5,
                'petal_width': 2.0
            }
        }
        
        # Generate data for each cluster
        data = []
        samples_per_cluster = n_samples // 3
        
        for species, centers in cluster_centers.items():
            for _ in range(samples_per_cluster):
                # Add some noise to make it realistic
                sepal_length = np.random.normal(centers['sepal_length'], 0.35)
                sepal_width = np.random.normal(centers['sepal_width'], 0.4)
                petal_length = np.random.normal(centers['petal_length'], 0.47)
                petal_width = np.random.normal(centers['petal_width'], 0.17)
                
                data.append({
                    'sepal_length': sepal_length,
                    'sepal_width': sepal_width,
                    'petal_length': petal_length,
                    'petal_width': petal_width,
                    'species': species
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Generated synthetic data with {len(df)} samples")
        return df
    
    def load_data(self):
        """
        Load Iris dataset (either real or synthetic)
        """
        try:
            if self.use_synthetic:
                logger.info("Loading synthetic Iris dataset")
                self.raw_data = self.generate_synthetic_iris()
            else:
                logger.info("Loading real Iris dataset from scikit-learn")
                iris = load_iris()
                self.raw_data = pd.DataFrame(
                    iris.data,
                    columns=iris.feature_names
                )
                self.raw_data['species'] = iris.target_names[iris.target]
            
            logger.info(f"Data loaded successfully: {self.raw_data.shape}")
            logger.info(f"Columns: {list(self.raw_data.columns)}")
            
            # Separate features and target
            self.features = self.raw_data.drop('species', axis=1)
            self.target = self.raw_data['species']
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def check_missing_values(self):
        """
        Check for missing values in the dataset
        """
        logger.info("Checking for missing values")
        
        missing_counts = self.raw_data.isnull().sum()
        missing_percentages = (missing_counts / len(self.raw_data)) * 100
        
        missing_summary = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing_Count': missing_counts.values,
            'Missing_Percentage': missing_percentages.values
        })
        
        logger.info("Missing values summary:")
        logger.info(missing_summary)
        
        return missing_summary
    
    def handle_missing_values(self):
        """
        Handle any missing values in the dataset
        """
        logger.info("Handling missing values")
        
        # Check if there are any missing values
        missing_before = self.raw_data.isnull().sum().sum()
        
        if missing_before > 0:
            # For numerical columns, fill with median
            numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.raw_data[col].isnull().sum() > 0:
                    median_val = self.raw_data[col].median()
                    self.raw_data[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
            
            # For categorical columns, fill with mode
            categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.raw_data[col].isnull().sum() > 0:
                    mode_val = self.raw_data[col].mode()[0]
                    self.raw_data[col].fillna(mode_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        else:
            logger.info("No missing values found in the dataset")
        
        missing_after = self.raw_data.isnull().sum().sum()
        logger.info(f"Missing values handled: {missing_before - missing_after} values processed")
    
    def normalize_features(self):
        """
        Normalize features using Min-Max scaling
        """
        logger.info("Normalizing features using Min-Max scaling")
        
        # Apply Min-Max scaling to features
        features_normalized = self.scaler.fit_transform(self.features)
        
        # Create new DataFrame with normalized features
        self.processed_data = pd.DataFrame(
            features_normalized,
            columns=self.features.columns
        )
        self.processed_data['species'] = self.target
        
        logger.info("Feature normalization completed")
        logger.info(f"Normalized data shape: {self.processed_data.shape}")
        
        return self.processed_data
    
    def explore_data(self):
        """
        Explore the dataset with summary statistics and visualizations
        """
        logger.info("Starting data exploration")
        
        # 1. Summary statistics
        logger.info("Computing summary statistics")
        summary_stats = self.raw_data.describe()
        logger.info("Summary statistics:")
        logger.info(summary_stats)
        
        # Save summary statistics
        summary_stats.to_csv('../data/iris_summary_statistics.csv')
        
        # 2. Species distribution
        species_counts = self.raw_data['species'].value_counts()
        logger.info("Species distribution:")
        logger.info(species_counts)
        
        # 3. Create visualizations
        self.create_visualizations()
        
        return summary_stats
    
    def create_visualizations(self):
        """
        Create various visualizations for data exploration
        """
        logger.info("Creating visualizations")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Pairplot
        logger.info("Creating pairplot")
        pairplot = sns.pairplot(
            self.raw_data,
            hue='species',
            diag_kind='hist',
            height=2.5
        )
        pairplot.fig.suptitle('Iris Dataset - Feature Relationships', y=1.02, size=16)
        pairplot.fig.tight_layout()
        plt.savefig('../images/iris_pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap
        logger.info("Creating correlation heatmap")
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.raw_data.drop('species', axis=1).corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5
        )
        plt.title('Iris Dataset - Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('../images/iris_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Boxplots for outlier detection
        logger.info("Creating boxplots for outlier detection")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Iris Dataset - Feature Distributions and Outliers', fontsize=16, y=0.95)
        
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for i, feature in enumerate(features):
            row = i // 2
            col = i % 2
            
            # Create boxplot
            sns.boxplot(data=self.raw_data, x='species', y=feature, ax=axes[row, col])
            axes[row, col].set_title(f'{feature.replace("_", " ").title()}')
            axes[row, col].set_xlabel('Species')
            axes[row, col].set_ylabel(feature.replace("_", " ").title())
        
        plt.tight_layout()
        plt.savefig('../images/iris_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Histograms for each feature
        logger.info("Creating histograms")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Iris Dataset - Feature Distributions by Species', fontsize=16, y=0.95)
        
        for i, feature in enumerate(features):
            row = i // 2
            col = i % 2
            
            for species in self.raw_data['species'].unique():
                species_data = self.raw_data[self.raw_data['species'] == species][feature]
                axes[row, col].hist(species_data, alpha=0.7, label=species, bins=15)
            
            axes[row, col].set_title(f'{feature.replace("_", " ").title()}')
            axes[row, col].set_xlabel(feature.replace("_", " ").title())
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig('../images/iris_histograms.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("All visualizations created and saved")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        logger.info(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
        
        if self.processed_data is None:
            logger.warning("No processed data available. Using raw data for splitting.")
            features = self.features
            target = self.target
        else:
            features = self.processed_data.drop('species', axis=1)
            target = self.processed_data['species']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Testing set: {X_test.shape[0]} samples")
        
        # Save split data
        train_data = X_train.copy()
        train_data['species'] = y_train
        test_data = X_test.copy()
        test_data['species'] = y_test
        
        train_data.to_csv('../data/iris_train.csv', index=False)
        test_data.to_csv('../data/iris_test.csv', index=False)
        
        logger.info("Train/test data saved to CSV files")
        
        return X_train, X_test, y_train, y_test
    
    def run_full_preprocessing(self):
        """
        Run the complete preprocessing pipeline
        """
        logger.info("Starting full preprocessing pipeline")
        
        try:
            # 1. Load data
            logger.info("=== LOADING PHASE ===")
            self.load_data()
            
            # 2. Check missing values
            logger.info("=== MISSING VALUES CHECK ===")
            missing_summary = self.check_missing_values()
            
            # 3. Handle missing values
            logger.info("=== MISSING VALUES HANDLING ===")
            self.handle_missing_values()
            
            # 4. Explore data
            logger.info("=== EXPLORATION PHASE ===")
            summary_stats = self.explore_data()
            
            # 5. Normalize features
            logger.info("=== NORMALIZATION PHASE ===")
            self.normalize_features()
            
            # 6. Split data
            logger.info("=== DATA SPLITTING ===")
            X_train, X_test, y_train, y_test = self.split_data()
            
            # 7. Save processed data
            logger.info("=== SAVING PHASE ===")
            if self.processed_data is not None:
                self.processed_data.to_csv('../data/iris_processed.csv', index=False)
                logger.info("Processed data saved to CSV")
            
            # Summary
            logger.info("=== PREPROCESSING SUMMARY ===")
            logger.info(f"Raw data shape: {self.raw_data.shape}")
            if self.processed_data is not None:
                logger.info(f"Processed data shape: {self.processed_data.shape}")
            logger.info(f"Training samples: {len(X_train)}")
            logger.info(f"Testing samples: {len(X_test)}")
            logger.info(f"Features: {list(self.features.columns)}")
            logger.info(f"Target classes: {list(self.target.unique())}")
            
            logger.info("Preprocessing pipeline completed successfully!")
            
            return {
                'raw_data': self.raw_data,
                'processed_data': self.processed_data,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'summary_stats': summary_stats
            }
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the preprocessing pipeline"""
    try:
        # Initialize preprocessor (set use_synthetic=True to generate synthetic data)
        preprocessor = IrisPreprocessor(use_synthetic=True)
        
        # Run full preprocessing pipeline
        results = preprocessor.run_full_preprocessing()
        
        logger.info("Preprocessing completed successfully!")
        logger.info("Check the 'images' and 'data' directories for outputs")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
