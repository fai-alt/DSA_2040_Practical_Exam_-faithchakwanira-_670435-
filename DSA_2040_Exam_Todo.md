# DSA 2040 US 2025 End Semester Exam - Practical Exam Todo List

**Exam Type:** Practical  
**Total Marks:** 100  
**Subject:** Data Warehousing and Data Mining  

## 📋 Pre-Exam Setup Checklist

- [x] **Environment Setup**
  - [x] Install Python 3.x ✅ (Python 3.11.9)
  - [x] Install required libraries: pandas, numpy, scikit-learn, sqlite3 ✅
  - [x] Install additional libraries: matplotlib, seaborn, mlxtend ✅
  - [x] Install SQL client (DB Browser for SQLite) ⚠️ (SQLite3 available through Python)
  - [x] Test all installations ✅

- [x] **Repository Setup**
  - [x] Create GitHub repository: "DSA_2040_Practical_Exam_[YourName_LastThreeDigitsID]" ✅
  - [x] Clone repository locally ✅
  - [x] Create project structure ✅

---

## 🏗️ Section 1: Data Warehousing (50 Marks)

### Task 1: Data Warehouse Design (15 Marks)

- [x] **Schema Design (8 marks)**
  - [x] Design star schema with 1 fact table + 4 dimension tables
  - [x] Define fact table measures (sales amount, quantity) and foreign keys
  - [x] Define dimension table attributes (customer, product, time, location dimensions)
  - [x] Create schema diagram using markdown and documentation
  - [x] Export diagram as documentation file

- [x] **Documentation (3 marks)**
  - [x] Write explanation in README.md: Why star schema over snowflake (comprehensive explanation)

- [x] **SQL Implementation (4 marks)**
  - [x] Write CREATE TABLE statements for fact table
  - [x] Write CREATE TABLE statements for dimension tables
  - [x] Use SQLite syntax
  - [x] Save as .sql file

**Deliverables:** Schema diagram image, README.md explanation, SQL script file

---

### Task 2: ETL Process Implementation (20 Marks)

- [x] **Dataset Preparation**
  - [x] Choose: UCI Online Retail dataset OR generate synthetic data ✅
  - [x] If synthetic: Generate ~1000 rows with similar structure ✅
  - [x] Include columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country ✅

- [x] **Extract (8 marks)**
  - [x] Write Python code to read CSV or generate DataFrame
  - [x] Handle missing values
  - [x] Convert InvoiceDate to datetime
  - [x] Handle data types appropriately

- [x] **Transform (8 marks)**
  - [x] Calculate TotalSales = Quantity * UnitPrice
  - [x] Create customer summary (group by CustomerID)
  - [x] Filter for last year sales (assume August 12, 2025 as current date)
  - [x] Remove outliers: Quantity < 0 or UnitPrice <= 0

- [x] **Load (6 marks)**
  - [x] Use sqlite3 to create retail_dw.db
  - [x] Load data into SalesFact table
  - [x] Load data into CustomerDim table
  - [x] Load data into TimeDim table

- [x] **Functionality (4 marks)**
  - [x] Write function to perform full ETL
  - [x] Log number of rows processed at each stage
  - [x] Include error handling

- [x] **Documentation (2 marks)**
  - [x] Add comprehensive comments
  - [x] Include logging functionality

**Deliverables:** Python script (etl_retail.py), .db file, generated CSV (if applicable), screenshots of table contents

---

### Task 3: OLAP Queries and Analysis (15 Marks)

- [x] **SQL Queries (6 marks)**
  - [x] Roll-up: Total sales by country and quarter
  - [x] Drill-down: Sales details for specific country by month
  - [x] Slice: Total sales for electronics category

- [x] **Visualization (4 marks)**
  - [x] Use Python (pandas/matplotlib) to visualize one query result
  - [x] Create bar chart of sales by country
  - [x] Save visualization as image

- [x] **Analysis Report (5 marks)**
  - [x] Write comprehensive report discussing insights
  - [x] Analyze top-selling countries and trends
  - [x] Explain how warehouse supports decision-making
  - [x] Note impact of synthetic data if used

**Deliverables:** SQL queries file, visualization image, analysis report (PDF/Markdown)

---

## 🧠 Section 2: Data Mining (50 Marks)

### Task 1: Data Preprocessing and Exploration (15 Marks)

- [x] **Dataset Preparation**
  - [x] Choose: scikit-learn Iris dataset OR generate synthetic data ✅
  - [x] If synthetic: Generate 150 samples with 3 clusters mimicking species ✅
  - [x] Include features: sepal length/width, petal length/width, class ✅

- [x] **Loading and Preprocessing (6 marks)**
  - [x] Load dataset using pandas/scikit-learn (or generate)
  - [x] Handle missing values (demonstrate checks)
  - [x] Normalize features using Min-Max scaling
  - [x] Encode class labels if needed

- [x] **Exploration and Visualization (6 marks)**
  - [x] Compute summary statistics using pandas.describe()
  - [x] Create pairplot using seaborn
  - [x] Create correlation heatmap
  - [x] Identify outliers using boxplots

- [x] **Data Splitting (3 marks)**
  - [x] Write function to split data into train/test (80/20)

**Deliverables:** Python script (preprocessing_iris.py), visualizations as images, generated data code/CSV (if applicable)

---

### Task 2: Clustering (15 Marks)

- [x] **Implementation and Metrics (7 marks)**
  - [x] Apply K-Means clustering with k=3
  - [x] Fit model on features (exclude class)
  - [x] Predict clusters and compare with actual classes using Adjusted Rand Index (ARI)

- [x] **Experimentation and Visualization (4 marks)**
  - [x] Try k=2 and k=4
  - [x] Plot elbow curve to justify optimal k
  - [x] Visualize clusters (scatter plot of petal length vs width, colored by cluster)

- [x] **Analysis (4 marks)**
  - [x] Write comprehensive analysis discussing cluster quality
  - [x] Analyze misclassifications
  - [x] Discuss real-world applications (e.g., customer segmentation)
  - [x] Note impact of synthetic data if used

**Deliverables:** Python script (clustering_iris.py), visualization images, analysis in report

---

### Task 3: Classification and Association Rule Mining (20 Marks)

#### Part A: Classification (10 Marks)

- [x] **Implementation and Metrics (5 marks)**
  - [x] Train Decision Tree classifier on train set
  - [x] Predict on test set
  - [x] Compute accuracy, precision, recall, F1-score

- [x] **Comparison and Visualization (5 marks)**
  - [x] Compare with another classifier (e.g., KNN with k=5)
  - [x] Visualize the tree using plot_tree
  - [x] Report which classifier is better and why

#### Part B: Association Rule Mining (10 Marks)

- [x] **Data Generation and Implementation (5 marks)**
  - [x] Generate synthetic transactional data (50 transactions) ✅
  - [x] Each transaction: 3-8 items from pool of 20 items ✅
  - [x] Apply Apriori algorithm (use mlxtend library) ✅
  - [x] Find rules with min_support=0.2, min_confidence=0.5 ✅
  - [x] Sort by lift and display top 5 rules ✅

- [x] **Rule Analysis (5 marks)**
  - [x] Discuss rule implications for retail recommendations

**Deliverables:** Python script (mining_iris_basket.py), outputs, generated data, analysis

---

## 📁 Final Submission Checklist

- [x] **Code Files**
  - [x] etl_retail.py
  - [x] preprocessing_iris.py
  - [x] clustering_iris.py
  - [x] classification_iris.py
  - [x] mining_iris_basket.py

- [x] **Data Files**
  - [x] retail_dw.db
  - [x] Generated CSV files (if applicable)
  - [x] SQL scripts

- [x] **Documentation**
  - [x] README.md with overview and instructions
  - [x] Analysis reports
  - [x] Self-assessment of completion

- [x] **Visualizations and Outputs**
  - [x] Schema diagram
  - [x] Visualization images
  - [x] Screenshots of outputs
  - [x] Table contents screenshots

- [x] **Repository Organization**
  - [x] All files properly organized
  - [x] Clear file naming
  - [x] Comprehensive documentation
  - [x] Instructions for running code

---

## ⚠️ Important Notes

- **Plagiarism:** Zero tolerance - use your own work
- **Code Quality:** Must be well-commented, modular, and error-free
- **Testing:** Ensure all code runs without errors locally
- **Partial Credit:** Include all attempted work, even if incomplete
- **Reproducibility:** If generating data, use seeds for reproducibility
- **Documentation:** Include explanations in markdown cells or comments

---

## 🎯 Mark Allocation Summary

- **Data Warehousing (50 marks):**
  - Task 1: 15 marks
  - Task 2: 20 marks
  - Task 3: 15 marks

- **Data Mining (50 marks):**
  - Task 1: 15 marks
  - Task 2: 15 marks
  - Task 3: 20 marks

**Total: 100 marks**

---

## 🚀 Getting Started

1. Complete the pre-exam setup checklist
2. Work through tasks systematically
3. Test each component thoroughly
4. Document everything clearly
5. Submit complete repository with all deliverables

**Good luck with your exam! 🎓**
