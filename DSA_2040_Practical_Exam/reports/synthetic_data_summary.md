# Synthetic Data Summary Report
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
