#!/usr/bin/env python3
"""
Synthetic Retail Data Generator for DSA 2040 Practical Exam
Generates ~1000 rows of retail data similar to UCI Online Retail dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_retail_data(n_rows=1000):
    """
    Generate synthetic retail data with specified structure
    
    Args:
        n_rows (int): Number of rows to generate (default: 1000)
    
    Returns:
        pd.DataFrame: Synthetic retail data
    """
    
    # Product categories and descriptions
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys', 'Beauty', 'Automotive']
    
    # Sample product descriptions for each category
    product_descriptions = {
        'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Camera', 'Speaker', 'Charger', 'Cable'],
        'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Sweater', 'Jacket', 'Shoes', 'Hat', 'Scarf'],
        'Home & Garden': ['Lamp', 'Plant Pot', 'Garden Tool', 'Furniture', 'Kitchen Item', 'Decor'],
        'Sports': ['Basketball', 'Tennis Racket', 'Yoga Mat', 'Running Shoes', 'Water Bottle', 'Gym Bag'],
        'Books': ['Fiction Novel', 'Non-Fiction', 'Textbook', 'Magazine', 'Comic Book', 'Cookbook'],
        'Toys': ['Action Figure', 'Board Game', 'Puzzle', 'Doll', 'Car Toy', 'Building Blocks'],
        'Beauty': ['Lipstick', 'Foundation', 'Mascara', 'Perfume', 'Skincare', 'Hair Product'],
        'Automotive': ['Car Accessory', 'Tool Kit', 'Cleaning Product', 'Maintenance Item']
    }
    
    # Countries for customer distribution
    countries = ['United Kingdom', 'Germany', 'France', 'EIRE', 'Spain', 'Netherlands', 'Belgium', 'Switzerland', 'Portugal', 'Italy']
    
    # Generate data
    data = []
    
    for i in range(n_rows):
        # Generate invoice number (some customers have multiple items per invoice)
        if i < n_rows * 0.7:  # 70% of rows are part of multi-item invoices
            invoice_no = f"INV{random.randint(1000, 9999)}"
        else:
            invoice_no = f"INV{random.randint(1000, 9999)}"
        
        # Generate stock code
        stock_code = f"SKU{random.randint(10000, 99999)}"
        
        # Select random category and description
        category = random.choice(categories)
        description = random.choice(product_descriptions[category])
        
        # Generate quantity (1-50, with some negative for returns)
        if random.random() < 0.05:  # 5% returns
            quantity = random.randint(-20, -1)
        else:
            quantity = random.randint(1, 50)
        
        # Generate invoice date (over 2 years, ending at August 12, 2025)
        end_date = datetime(2025, 8, 12)
        start_date = end_date - timedelta(days=2*365)
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        invoice_date = start_date + timedelta(days=random_days)
        
        # Generate unit price (1-100, with some variation by category)
        base_price = random.uniform(1, 100)
        if category == 'Electronics':
            base_price *= random.uniform(1.5, 3.0)  # Electronics are more expensive
        elif category == 'Books':
            base_price *= random.uniform(0.3, 0.8)  # Books are cheaper
        
        unit_price = round(base_price, 2)
        
        # Generate customer ID (100 unique customers)
        customer_id = random.randint(1000, 1099)
        
        # Select country (weighted towards UK as in original dataset)
        if random.random() < 0.4:  # 40% UK customers
            country = 'United Kingdom'
        else:
            country = random.choice(countries)
        
        # Create row
        row = {
            'InvoiceNo': invoice_no,
            'StockCode': stock_code,
            'Description': description,
            'Quantity': quantity,
            'InvoiceDate': invoice_date,
            'UnitPrice': unit_price,
            'CustomerID': customer_id,
            'Country': country
        }
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by invoice date for realistic ordering
    df = df.sort_values('InvoiceDate')
    
    return df

def main():
    """Main function to generate and save synthetic retail data"""
    
    print("Generating synthetic retail data for DSA 2040 Practical Exam...")
    
    # Generate data
    retail_df = generate_synthetic_retail_data(1000)
    
    # Display basic information
    print(f"\nGenerated {len(retail_df)} rows of synthetic retail data")
    print(f"Date range: {retail_df['InvoiceDate'].min()} to {retail_df['InvoiceDate'].max()}")
    print(f"Unique customers: {retail_df['CustomerID'].nunique()}")
    print(f"Unique products: {retail_df['StockCode'].nunique()}")
    print(f"Countries represented: {retail_df['Country'].nunique()}")
    
    # Display sample data
    print("\nSample data:")
    print(retail_df.head(10))
    
    # Display summary statistics
    print("\nSummary statistics:")
    print(retail_df.describe())
    
    # Save to CSV
    output_file = '../data/synthetic_retail_data.csv'
    retail_df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
    
    # Display data quality checks
    print("\nData quality checks:")
    print(f"Missing values: {retail_df.isnull().sum().sum()}")
    print(f"Negative quantities (returns): {len(retail_df[retail_df['Quantity'] < 0])}")
    print(f"Zero or negative prices: {len(retail_df[retail_df['UnitPrice'] <= 0])}")
    
    return retail_df

if __name__ == "__main__":
    retail_data = main()
