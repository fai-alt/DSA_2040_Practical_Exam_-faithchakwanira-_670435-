"""
DSA 2040 Practical Exam - Task 2: ETL Process Implementation
Data Warehousing Section - ETL for Retail Company

This script implements a complete ETL pipeline for retail data:
1. Extract: Read CSV or generate synthetic data
2. Transform: Clean, calculate, and prepare data
3. Load: Store in SQLite data warehouse

Author: [Your Name]
Date: August 13, 2025
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
import random
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_retail.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetailETL:
    def __init__(self, db_path='retail_dw.db'):
        """Initialize ETL process with database path"""
        self.db_path = db_path
        self.raw_data = None
        self.transformed_data = None
        self.customer_dim = None
        self.time_dim = None
        self.product_dim = None
        self.location_dim = None
        
    def generate_synthetic_data(self, num_rows=1000):
        """
        Generate synthetic retail data similar to Online Retail dataset
        Creates realistic patterns for testing and development
        """
        logger.info(f"Generating {num_rows} synthetic retail records")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Generate customer data
        countries = ['United Kingdom', 'Germany', 'France', 'Spain', 'Italy', 
                    'Netherlands', 'Belgium', 'Switzerland', 'Austria', 'Portugal']
        
        # Generate product categories
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        subcategories = {
            'Electronics': ['Smartphones', 'Laptops', 'Accessories', 'Gaming'],
            'Clothing': ['Men', 'Women', 'Kids', 'Accessories'],
            'Home & Garden': ['Furniture', 'Decor', 'Kitchen', 'Garden'],
            'Sports': ['Fitness', 'Outdoor', 'Team Sports', 'Equipment'],
            'Books': ['Fiction', 'Non-Fiction', 'Academic', 'Children']
        }
        
        # Generate data
        data = []
        start_date = datetime(2023, 8, 13)  # 2 years ago from exam date
        
        for i in range(num_rows):
            # Random date within 2 years
            random_days = random.randint(0, 730)
            invoice_date = start_date + timedelta(days=random_days)
            
            # Random customer
            customer_id = random.randint(1000, 1099)  # 100 unique customers
            
            # Random product
            category = random.choice(categories)
            subcategory = random.choice(subcategories[category])
            
            # Random quantities and prices
            quantity = random.randint(1, 50)
            unit_price = round(random.uniform(1.0, 100.0), 2)
            
            # Random country
            country = random.choice(countries)
            
            data.append({
                'InvoiceNo': f'INV{str(i+1).zfill(6)}',
                'StockCode': f'PROD{str(random.randint(1000, 9999))}',
                'Description': f'{subcategory} Product {i+1}',
                'Quantity': quantity,
                'InvoiceDate': invoice_date,
                'UnitPrice': unit_price,
                'CustomerID': customer_id,
                'Country': country
            })
        
        self.raw_data = pd.DataFrame(data)
        logger.info(f"Generated {len(self.raw_data)} synthetic records")
        return self.raw_data
    
    def extract_data(self, file_path=None):
        """
        Extract data from CSV file or generate synthetic data
        Args:
            file_path: Path to CSV file (optional, will generate if not provided)
        """
        try:
            if file_path and os.path.exists(file_path):
                logger.info(f"Loading data from {file_path}")
                self.raw_data = pd.read_csv(file_path)
                logger.info(f"Loaded {len(self.raw_data)} records from CSV")
            else:
                logger.info("No CSV file provided, generating synthetic data")
                self.generate_synthetic_data()
            
            # Convert InvoiceDate to datetime if it's not already
            if 'InvoiceDate' in self.raw_data.columns:
                self.raw_data['InvoiceDate'] = pd.to_datetime(self.raw_data['InvoiceDate'])
            
            logger.info(f"Extraction completed: {len(self.raw_data)} records")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            raise
    
    def transform_data(self):
        """
        Transform the raw data according to requirements:
        1. Calculate TotalSales = Quantity * UnitPrice
        2. Create customer summary extracts
        3. Filter for last year sales
        4. Handle outliers
        """
        logger.info("Starting data transformation")
        
        if self.raw_data is None:
            raise ValueError("No data to transform. Run extract_data() first.")
        
        # Make a copy to avoid modifying original data
        df = self.raw_data.copy()
        
        # 1. Calculate TotalSales
        df['TotalSales'] = df['Quantity'] * df['UnitPrice']
        logger.info("Calculated TotalSales column")
        
        # 2. Handle outliers: Remove rows where Quantity < 0 or UnitPrice <= 0
        initial_count = len(df)
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} outlier records")
        
        # 3. Filter for last year sales (assume current date as August 12, 2025)
        current_date = datetime(2025, 8, 12)
        one_year_ago = current_date - timedelta(days=365)
        df = df[df['InvoiceDate'] >= one_year_ago]
        logger.info(f"Filtered to last year: {len(df)} records")
        
        # 4. Handle missing values
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        missing_after = df.isnull().sum().sum()
        logger.info(f"Handled missing values: {missing_before - missing_after} values removed")
        
        # 5. Create customer summary extracts
        customer_summary = df.groupby('CustomerID').agg({
            'TotalSales': 'sum',
            'Quantity': 'sum',
            'Country': 'first',
            'InvoiceNo': 'nunique'
        }).reset_index()
        customer_summary.columns = ['CustomerID', 'TotalPurchases', 'TotalQuantity', 'Country', 'OrderCount']
        customer_summary['AverageOrderValue'] = customer_summary['TotalPurchases'] / customer_summary['OrderCount']
        
        self.customer_dim = customer_summary
        logger.info(f"Created customer dimension with {len(customer_summary)} customers")
        
        # 6. Create time dimension extracts
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
        
        time_summary = df.groupby(['Year', 'Quarter', 'Month']).agg({
            'TotalSales': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        self.time_dim = time_summary
        logger.info(f"Created time dimension with {len(time_summary)} time periods")
        
        # 7. Create product dimension extracts
        product_summary = df.groupby('StockCode').agg({
            'Description': 'first',
            'UnitPrice': 'mean',
            'Quantity': 'sum',
            'TotalSales': 'sum'
        }).reset_index()
        
        # Add category information (simplified)
        product_summary['Category'] = 'General'
        product_summary['SubCategory'] = 'Standard'
        
        self.product_dim = product_summary
        logger.info(f"Created product dimension with {len(product_summary)} products")
        
        # 8. Create location dimension extracts
        location_summary = df.groupby('Country').agg({
            'TotalSales': 'sum',
            'CustomerID': 'nunique',
            'InvoiceNo': 'nunique'
        }).reset_index()
        location_summary['Region'] = 'Europe'  # Simplified for synthetic data
        
        self.location_dim = location_summary
        logger.info(f"Created location dimension with {len(location_summary)} countries")
        
        self.transformed_data = df
        logger.info(f"Transformation completed: {len(df)} records")
        return df
    
    def load_to_database(self):
        """
        Load transformed data into SQLite database
        Creates tables and populates them with the transformed data
        """
        logger.info("Starting database load process")
        
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Read and execute SQL schema
            with open('sql/create_tables.sql', 'r') as f:
                sql_script = f.read()
                cursor.executescript(sql_script)
            
            logger.info("Database schema created successfully")
            
            # Load customer dimension
            if self.customer_dim is not None:
                self.customer_dim.to_sql('CustomerDim', conn, if_exists='replace', index=False)
                logger.info(f"Loaded {len(self.customer_dim)} customer records")
            
            # Load time dimension
            if self.time_dim is not None:
                # Create proper time dimension with TimeID
                time_dim_with_id = self.time_dim.copy()
                time_dim_with_id['TimeID'] = range(1, len(time_dim_with_id) + 1)
                time_dim_with_id['Date'] = pd.to_datetime(time_dim_with_id[['Year', 'Month']].assign(day=1))
                time_dim_with_id['FiscalYear'] = time_dim_with_id['Year'].astype(str)
                time_dim_with_id['IsHoliday'] = False  # Simplified
                
                time_dim_with_id.to_sql('TimeDim', conn, if_exists='replace', index=False)
                logger.info(f"Loaded {len(time_dim_with_id)} time records")
            
            # Load product dimension
            if self.product_dim is not None:
                # Create proper product dimension with ProductID
                product_dim_with_id = self.product_dim.copy()
                product_dim_with_id['ProductID'] = range(1, len(product_dim_with_id) + 1)
                product_dim_with_id['Brand'] = 'Generic'
                product_dim_with_id['Cost'] = product_dim_with_id['UnitPrice'] * 0.6  # Simplified cost
                product_dim_with_id['Supplier'] = 'Default Supplier'
                
                product_dim_with_id.to_sql('ProductDim', conn, if_exists='replace', index=False)
                logger.info(f"Loaded {len(product_dim_with_id)} product records")
            
            # Load location dimension
            if self.location_dim is not None:
                # Create proper location dimension with LocationID
                location_dim_with_id = self.location_dim.copy()
                location_dim_with_id['LocationID'] = range(1, len(location_dim_with_id) + 1)
                location_dim_with_id['City'] = 'Capital'  # Simplified
                location_dim_with_id['Population'] = 1000000  # Simplified
                location_dim_with_id['GDP'] = 1000000000  # Simplified
                
                location_dim_with_id.to_sql('LocationDim', conn, if_exists='replace', index=False)
                logger.info(f"Loaded {len(location_dim_with_id)} location records")
            
            # Load sales fact table
            if self.transformed_data is not None:
                # Create fact table records
                fact_records = []
                for idx, row in self.transformed_data.iterrows():
                    # Find corresponding dimension IDs
                    customer_id = row['CustomerID']
                    product_id = row['StockCode']
                    time_id = 1  # Simplified mapping
                    location_id = 1  # Simplified mapping
                    
                    fact_records.append({
                        'CustomerID': customer_id,
                        'ProductID': product_id,
                        'TimeID': time_id,
                        'LocationID': location_id,
                        'InvoiceNo': row['InvoiceNo'],
                        'Quantity': row['Quantity'],
                        'UnitPrice': row['UnitPrice'],
                        'TotalSales': row['TotalSales'],
                        'Discount': 0.0,  # Simplified
                        'NetSales': row['TotalSales']
                    })
                
                fact_df = pd.DataFrame(fact_records)
                fact_df.to_sql('SalesFact', conn, if_exists='replace', index=False)
                logger.info(f"Loaded {len(fact_df)} sales fact records")
            
            conn.commit()
            conn.close()
            logger.info("Database load completed successfully")
            
        except Exception as e:
            logger.error(f"Error during database load: {str(e)}")
            raise
    
    def run_full_etl(self, csv_file=None):
        """
        Run the complete ETL pipeline
        Args:
            csv_file: Optional path to CSV file
        """
        logger.info("Starting full ETL pipeline")
        
        try:
            # Extract
            logger.info("=== EXTRACT PHASE ===")
            self.extract_data(csv_file)
            logger.info(f"Extracted {len(self.raw_data)} records")
            
            # Transform
            logger.info("=== TRANSFORM PHASE ===")
            self.transform_data()
            logger.info(f"Transformed {len(self.transformed_data)} records")
            
            # Load
            logger.info("=== LOAD PHASE ===")
            self.load_to_database()
            logger.info("Data loaded to database successfully")
            
            # Summary
            logger.info("=== ETL SUMMARY ===")
            logger.info(f"Raw data records: {len(self.raw_data)}")
            logger.info(f"Transformed records: {len(self.transformed_data)}")
            logger.info(f"Customer dimension: {len(self.customer_dim)} records")
            logger.info(f"Product dimension: {len(self.product_dim)} records")
            logger.info(f"Time dimension: {len(self.time_dim)} records")
            logger.info(f"Location dimension: {len(self.location_dim)} records")
            
            logger.info("ETL pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the ETL process"""
    try:
        # Initialize ETL process
        etl = RetailETL()
        
        # Run full ETL pipeline
        etl.run_full_etl()
        
        # Save transformed data to CSV for inspection
        if etl.transformed_data is not None:
            etl.transformed_data.to_csv('../data/transformed_retail_data.csv', index=False)
            logger.info("Transformed data saved to CSV")
        
        # Save dimension tables to CSV for inspection
        if etl.customer_dim is not None:
            etl.customer_dim.to_csv('../data/customer_dimension.csv', index=False)
        if etl.product_dim is not None:
            etl.product_dim.to_csv('../data/product_dimension.csv', index=False)
        if etl.time_dim is not None:
            etl.time_dim.to_csv('../data/time_dimension.csv', index=False)
        if etl.location_dim is not None:
            etl.location_dim.to_csv('../data/location_dimension.csv', index=False)
        
        logger.info("All data files saved successfully")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
