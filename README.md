# DSA 2040 Practical Exam: Data Warehousing and Data Mining

**Exam Type:** Practical  
**Total Marks:** 100  
**Student:** [Your Name]  
**ID:** [Your ID]  
**Date:** August 13, 2025

## Project Overview

This repository contains the complete implementation for the DSA 2040 End Semester Practical Exam covering both Data Warehousing and Data Mining sections.

## Project Structure

```
DSA_2040_Practical_Exam/
├── data/                   # Datasets and generated data
├── src/                    # Python source code files
├── docs/                   # Documentation and schema diagrams
├── outputs/                # Generated outputs and results
├── sql/                    # SQL scripts and database files
├── images/                 # Visualizations and charts
├── reports/                # Analysis reports and findings
└── README.md               # This file
```

## Section 1: Data Warehousing (50 Marks)

### Task 1: Data Warehouse Design (15 Marks)
- **Schema Design and Diagram:** 8 marks
- **Explanation:** 3 marks  
- **SQL Statements:** 4 marks

**Deliverables:**
- Schema diagram (image file)
- SQL CREATE TABLE statements
- Explanation of star schema choice

### Task 2: ETL Process Implementation (20 Marks)
- **Extraction and Transformation:** 8 marks
- **Loading to Database:** 6 marks
- **Functionality and Error Handling:** 4 marks
- **Comments and Logging:** 2 marks

**Deliverables:**
- Python ETL script (etl_retail.py)
- Database file (retail_dw.db)
- Generated/processed datasets

### Task 3: OLAP Queries and Analysis (15 Marks)
- **SQL Queries:** 6 marks
- **Visualization:** 4 marks
- **Analysis Report:** 5 marks

**Deliverables:**
- OLAP SQL queries
- Visualization charts
- Analysis report

## Section 2: Data Mining (50 Marks)

### Task 1: Data Preprocessing and Exploration (15 Marks)
- **Loading and Preprocessing:** 6 marks
- **Exploration and Visualizations:** 6 marks
- **Data Split Function:** 3 marks

**Deliverables:**
- Preprocessing script (preprocessing_iris.py)
- Exploration visualizations
- Train/test split function

### Task 2: Clustering (15 Marks)
- **Implementation and Metrics:** 7 marks
- **Experimentation and Visualization:** 4 marks
- **Analysis:** 4 marks

**Deliverables:**
- Clustering script (clustering_iris.py)
- Cluster visualizations
- Analysis report

### Task 3: Classification and Association Rule Mining (20 Marks)
- **Classification Implementation:** 10 marks
- **Association Rule Mining:** 10 marks

**Deliverables:**
- Mining script (mining_iris_basket.py)
- Classification results
- Association rules analysis

## Datasets Used

### Data Warehousing
- **Option A:** Online Retail dataset from UCI ML Repository
- **Option B:** Synthetic retail data (1000 rows, similar structure)

### Data Mining
- **Option A:** Iris dataset from scikit-learn
- **Option B:** Synthetic iris-like data (150 samples, 3 clusters)

## Prerequisites

- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - sqlite3
  - matplotlib
  - seaborn
  - mlxtend

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend
```

## How to Run

1. **Data Warehousing Section:**
   ```bash
   python src/etl_retail.py
   ```

2. **Data Mining Section:**
   ```bash
   python src/preprocessing_iris.py
   python src/clustering_iris.py
   python src/mining_iris_basket.py
   ```

## Self-Assessment

[To be completed after implementation]

## Notes

- All code includes comprehensive error handling and logging
- Synthetic data generation uses reproducible random seeds
- Visualizations are saved as image files for submission
- Database files and outputs are included where applicable

## Submission Checklist

- [ ] All Python scripts (.py files)
- [ ] SQL scripts and database files
- [ ] Generated datasets (CSV files)
- [ ] Visualization images
- [ ] Analysis reports
- [ ] Schema diagrams
- [ ] README.md with complete overview
- [ ] All code runs without errors
- [ ] Comprehensive documentation and comments
