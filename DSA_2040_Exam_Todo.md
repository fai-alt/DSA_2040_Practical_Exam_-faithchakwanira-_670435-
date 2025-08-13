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

- [ ] **Schema Design (8 marks)**
  - [ ] Design star schema with 1 fact table + 3-4 dimension tables
  - [ ] Define fact table measures (sales amount, quantity) and foreign keys
  - [ ] Define dimension table attributes (customer, product, time dimensions)
  - [ ] Create schema diagram using Draw.io or hand-drawn
  - [ ] Export diagram as image file

- [ ] **Documentation (3 marks)**
  - [ ] Write explanation in README.md: Why star schema over snowflake (2-3 sentences)

- [ ] **SQL Implementation (4 marks)**
  - [ ] Write CREATE TABLE statements for fact table
  - [ ] Write CREATE TABLE statements for dimension tables
  - [ ] Use SQLite syntax
  - [ ] Save as .sql file

**Deliverables:** Schema diagram image, README.md explanation, SQL script file

---

### Task 2: ETL Process Implementation (20 Marks)

- [ ] **Dataset Preparation**
  - [ ] Choose: UCI Online Retail dataset OR generate synthetic data
  - [ ] If synthetic: Generate ~1000 rows with similar structure
  - [ ] Include columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

- [ ] **Extract (8 marks)**
  - [ ] Write Python code to read CSV or generate DataFrame
  - [ ] Handle missing values
  - [ ] Convert InvoiceDate to datetime
  - [ ] Handle data types appropriately

- [ ] **Transform (8 marks)**
  - [ ] Calculate TotalSales = Quantity * UnitPrice
  - [ ] Create customer summary (group by CustomerID)
  - [ ] Filter for last year sales (assume August 12, 2025 as current date)
  - [ ] Remove outliers: Quantity < 0 or UnitPrice <= 0

- [ ] **Load (6 marks)**
  - [ ] Use sqlite3 to create retail_dw.db
  - [ ] Load data into SalesFact table
  - [ ] Load data into CustomerDim table
  - [ ] Load data into TimeDim table

- [ ] **Functionality (4 marks)**
  - [ ] Write function to perform full ETL
  - [ ] Log number of rows processed at each stage
  - [ ] Include error handling

- [ ] **Documentation (2 marks)**
  - [ ] Add comprehensive comments
  - [ ] Include logging functionality

**Deliverables:** Python script (etl_retail.py), .db file, generated CSV (if applicable), screenshots of table contents

---

### Task 3: OLAP Queries and Analysis (15 Marks)

- [ ] **SQL Queries (6 marks)**
  - [ ] Roll-up: Total sales by country and quarter
  - [ ] Drill-down: Sales details for specific country by month
  - [ ] Slice: Total sales for electronics category

- [ ] **Visualization (4 marks)**
  - [ ] Use Python (pandas/matplotlib) to visualize one query result
  - [ ] Create bar chart of sales by country
  - [ ] Save visualization as image

- [ ] **Analysis Report (5 marks)**
  - [ ] Write 200-300 word report discussing insights
  - [ ] Analyze top-selling countries and trends
  - [ ] Explain how warehouse supports decision-making
  - [ ] Note impact of synthetic data if used

**Deliverables:** SQL queries file, visualization image, analysis report (PDF/Markdown)

---

## 🧠 Section 2: Data Mining (50 Marks)

### Task 1: Data Preprocessing and Exploration (15 Marks)

- [ ] **Dataset Preparation**
  - [ ] Choose: scikit-learn Iris dataset OR generate synthetic data
  - [ ] If synthetic: Generate 150 samples with 3 clusters mimicking species
  - [ ] Include features: sepal length/width, petal length/width, class

- [ ] **Loading and Preprocessing (6 marks)**
  - [ ] Load dataset using pandas/scikit-learn (or generate)
  - [ ] Handle missing values (demonstrate checks)
  - [ ] Normalize features using Min-Max scaling
  - [ ] Encode class labels if needed

- [ ] **Exploration and Visualization (6 marks)**
  - [ ] Compute summary statistics using pandas.describe()
  - [ ] Create pairplot using seaborn
  - [ ] Create correlation heatmap
  - [ ] Identify outliers using boxplots

- [ ] **Data Splitting (3 marks)**
  - [ ] Write function to split data into train/test (80/20)

**Deliverables:** Python script (preprocessing_iris.py), visualizations as images, generated data code/CSV (if applicable)

---

### Task 2: Clustering (15 Marks)

- [ ] **Implementation and Metrics (7 marks)**
  - [ ] Apply K-Means clustering with k=3
  - [ ] Fit model on features (exclude class)
  - [ ] Predict clusters and compare with actual classes using Adjusted Rand Index (ARI)

- [ ] **Experimentation and Visualization (4 marks)**
  - [ ] Try k=2 and k=4
  - [ ] Plot elbow curve to justify optimal k
  - [ ] Visualize clusters (scatter plot of petal length vs width, colored by cluster)

- [ ] **Analysis (4 marks)**
  - [ ] Write 150-200 word analysis discussing cluster quality
  - [ ] Analyze misclassifications
  - [ ] Discuss real-world applications (e.g., customer segmentation)
  - [ ] Note impact of synthetic data if used

**Deliverables:** Python script (clustering_iris.py), visualization images, analysis in report

---

### Task 3: Classification and Association Rule Mining (20 Marks)

#### Part A: Classification (10 Marks)

- [ ] **Implementation and Metrics (5 marks)**
  - [ ] Train Decision Tree classifier on train set
  - [ ] Predict on test set
  - [ ] Compute accuracy, precision, recall, F1-score

- [ ] **Comparison and Visualization (5 marks)**
  - [ ] Compare with another classifier (e.g., KNN with k=5)
  - [ ] Visualize the tree using plot_tree
  - [ ] Report which classifier is better and why

#### Part B: Association Rule Mining (10 Marks)

- [ ] **Data Generation and Implementation (5 marks)**
  - [ ] Generate synthetic transactional data (20-50 transactions)
  - [ ] Each transaction: 3-8 items from pool of 20 items
  - [ ] Apply Apriori algorithm (use mlxtend library)
  - [ ] Find rules with min_support=0.2, min_confidence=0.5
  - [ ] Sort by lift and display top 5 rules

- [ ] **Rule Analysis (5 marks)**
  - [ ] Discuss one rule's implications for retail recommendations

**Deliverables:** Python script (mining_iris_basket.py), outputs, generated data, analysis

---

## 📁 Final Submission Checklist

- [ ] **Code Files**
  - [ ] etl_retail.py
  - [ ] preprocessing_iris.py
  - [ ] clustering_iris.py
  - [ ] mining_iris_basket.py

- [ ] **Data Files**
  - [ ] retail_dw.db
  - [ ] Generated CSV files (if applicable)
  - [ ] SQL scripts

- [ ] **Documentation**
  - [ ] README.md with overview and instructions
  - [ ] Analysis reports
  - [ ] Self-assessment of completion

- [ ] **Visualizations and Outputs**
  - [ ] Schema diagram
  - [ ] Visualization images
  - [ ] Screenshots of outputs
  - [ ] Table contents screenshots

- [ ] **Repository Organization**
  - [ ] All files properly organized
  - [ ] Clear file naming
  - [ ] Comprehensive documentation
  - [ ] Instructions for running code

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
