#!/usr/bin/env python3
"""
Master Data Generator for DSA 2040 Practical Exam
Generates all synthetic datasets required for the exam
"""

import os
import sys
import pandas as pd
import numpy as np

# Add current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_retail_data import generate_synthetic_retail_data
from generate_iris_data import generate_synthetic_iris_data, create_iris_dataframe
from generate_transactional_data import generate_synthetic_transactions, create_transaction_dataframe

def ensure_directories_exist():
    """Ensure all required directories exist"""
    
    directories = [
        '../data',
        '../images', 
        '../outputs',
        '../reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory ensured: {directory}")

def generate_retail_dataset():
    """Generate synthetic retail data for Data Warehousing section"""
    
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC RETAIL DATA")
    print("="*60)
    
    # Generate retail data
    retail_df = generate_synthetic_retail_data(1000)
    
    # Save to CSV
    output_file = '../data/synthetic_retail_data.csv'
    retail_df.to_csv(output_file, index=False)
    print(f"✓ Retail data saved to: {output_file}")
    
    # Display summary
    print(f"✓ Generated {len(retail_df)} rows of retail data")
    print(f"✓ Date range: {retail_df['InvoiceDate'].min()} to {retail_df['InvoiceDate'].max()}")
    print(f"✓ Unique customers: {retail_df['CustomerID'].nunique()}")
    print(f"✓ Countries: {retail_df['Country'].nunique()}")
    
    return retail_df

def generate_iris_dataset():
    """Generate synthetic Iris data for Data Mining section"""
    
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC IRIS DATA")
    print("="*60)
    
    # Generate Iris data
    features, labels, feature_names = generate_synthetic_iris_data(150, 3)
    
    # Create DataFrame
    iris_df = create_iris_dataframe(features, labels, feature_names)
    
    # Save to CSV
    output_file = '../data/synthetic_iris_data.csv'
    iris_df.to_csv(output_file, index=False)
    print(f"✓ Iris data saved to: {output_file}")
    
    # Display summary
    print(f"✓ Generated {len(iris_df)} samples of Iris data")
    print(f"✓ Features: {feature_names}")
    print(f"✓ Species distribution:")
    print(iris_df['species'].value_counts())
    
    return iris_df

def generate_transactional_dataset():
    """Generate synthetic transactional data for Association Rule Mining"""
    
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC TRANSACTIONAL DATA")
    print("="*60)
    
    # Generate transactional data
    transactions = generate_synthetic_transactions(40, 3, 8)
    
    # Create DataFrame
    df = create_transaction_dataframe(transactions)
    
    # Save to CSV
    csv_output = '../data/synthetic_transactions.csv'
    df.to_csv(csv_output, index=False)
    print(f"✓ Transaction data saved to: {csv_output}")
    
    # Save to readable text file
    txt_output = '../data/synthetic_transactions.txt'
    with open(txt_output, 'w') as f:
        f.write("# Synthetic Transactional Data for DSA 2040 Practical Exam\n")
        f.write(f"# Generated {len(transactions)} transactions\n\n")
        for i, transaction in enumerate(transactions):
            f.write(f"Transaction_{i+1}: {','.join(transaction)}\n")
    print(f"✓ Transaction data saved to: {txt_output}")
    
    # Display summary
    print(f"✓ Generated {len(transactions)} transactions")
    print(f"✓ Items per transaction: 3-8")
    print(f"✓ Total items: {sum(len(t) for t in transactions)}")
    
    return transactions, df

def create_data_summary_report():
    """Create a summary report of all generated datasets"""
    
    print("\n" + "="*60)
    print("CREATING DATA SUMMARY REPORT")
    print("="*60)
    
    report_content = """# Synthetic Data Summary Report
## DSA 2040 Practical Exam - Data Generation

### 1. Retail Dataset (Data Warehousing Section)
- **Purpose**: ETL Process Implementation and OLAP Analysis
- **Rows**: 1000
- **Columns**: 8 (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country)
- **Date Range**: 2 years ending August 12, 2025
- **Features**: 
  - Realistic product categories (Electronics, Clothing, Home & Garden, etc.)
  - 100 unique customers across 10 countries
  - Price variations by category
  - Some negative quantities (returns) for realism
- **File**: `data/synthetic_retail_data.csv`

### 2. Iris Dataset (Data Mining Section)
- **Purpose**: Clustering, Classification, and Data Exploration
- **Samples**: 150
- **Features**: 4 (sepal_length, sepal_width, petal_length, petal_width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Characteristics**: 
  - Realistic feature ranges matching actual Iris dataset
  - Clear cluster separation for effective analysis
  - Gaussian distributions with realistic standard deviations
- **File**: `data/synthetic_iris_data.csv`

### 3. Transactional Dataset (Association Rule Mining)
- **Purpose**: Apriori Algorithm Implementation
- **Transactions**: 40
- **Items per Transaction**: 3-8
- **Item Pool**: 20 common retail products
- **Features**:
  - Realistic co-occurrence patterns (milk+bread, beer+snacks, etc.)
  - Variable transaction sizes
  - Common retail behavior simulation
- **Files**: 
  - `data/synthetic_transactions.csv` (structured format)
  - `data/synthetic_transactions.txt` (readable format)

### Data Quality Features
- **Reproducibility**: All datasets use fixed random seeds (42)
- **Realism**: Data mimics real-world patterns and distributions
- **Consistency**: Appropriate data types and value ranges
- **Completeness**: No missing values, ready for immediate analysis

### Usage Instructions
1. **Retail Data**: Use for ETL pipeline, database loading, and OLAP queries
2. **Iris Data**: Use for clustering, classification, and data exploration
3. **Transactional Data**: Use for association rule mining with Apriori algorithm

### Technical Notes
- All data generated using Python 3.11.9
- Libraries: pandas, numpy, scikit-learn
- Random seeds ensure reproducible results across runs
- Data formats optimized for exam requirements

---
*Generated for DSA 2040 US 2025 End Semester Practical Exam*
*Data Warehousing and Data Mining - Total Marks: 100*
"""
    
    # Save report
    report_file = '../reports/synthetic_data_summary.md'
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"✓ Data summary report saved to: {report_file}")
    
    return report_content

def main():
    """Main function to generate all synthetic datasets"""
    
    print("🚀 DSA 2040 PRACTICAL EXAM - SYNTHETIC DATA GENERATOR")
    print("="*60)
    print("This script will generate all required synthetic datasets for your exam.")
    print("="*60)
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Generate all datasets
    retail_df = generate_retail_dataset()
    iris_df = generate_iris_dataset()
    transactions, trans_df = generate_transactional_dataset()
    
    # Create summary report
    create_data_summary_report()
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 ALL SYNTHETIC DATA GENERATED SUCCESSFULLY!")
    print("="*60)
    print("✓ Retail Dataset: 1000 rows for Data Warehousing")
    print("✓ Iris Dataset: 150 samples for Data Mining")
    print("✓ Transactional Dataset: 40 transactions for Association Rules")
    print("✓ Summary Report: Complete documentation")
    print("\n📁 Files saved in:")
    print("  - data/ directory: All CSV datasets")
    print("  - reports/ directory: Data summary report")
    print("\n🚀 You're now ready to begin the actual exam tasks!")
    print("="*60)
    
    return {
        'retail': retail_df,
        'iris': iris_df,
        'transactions': transactions,
        'transaction_df': trans_df
    }

if __name__ == "__main__":
    all_data = main()
