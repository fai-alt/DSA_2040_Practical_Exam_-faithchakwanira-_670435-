
# Classification Analysis Report
## DSA 2040 Practical Exam - Task 3 Part A

**Generated on:** 2025-08-14 18:01:43

### Executive Summary
This report analyzes the performance of Decision Tree and K-Nearest Neighbors (KNN) classifiers on the Iris dataset.

### Dataset Information
- **Dataset:** Iris dataset (processed)
- **Features:** 4 numerical features (sepal length/width, petal length/width)
- **Classes:** 3 species (setosa, versicolor, virginica)
- **Training Set:** 120 samples
- **Testing Set:** 30 samples

### Classifier Performance Comparison

| Metric | Decision Tree | KNN (k=5) |
|--------|---------------|------------|
| Accuracy | 1.0000 | 1.0000 |
| Precision (Macro) | 1.0000 | 1.0000 |
| Recall (Macro) | 1.0000 | 1.0000 |
| F1-Score (Macro) | 1.0000 | 1.0000 |


### Winner Analysis
**Best Classifier:** Tie
**Reason:** Both classifiers perform similarly

### Detailed Performance Analysis

#### Decision Tree Classifier
- **Configuration:** max_depth=5, min_samples_split=5, min_samples_leaf=2
- **Strengths:** 
  - Interpretable decision rules
  - Fast prediction time
  - Handles non-linear relationships
- **Weaknesses:**
  - Can overfit with complex trees
  - Sensitive to data variations

#### K-Nearest Neighbors Classifier
- **Configuration:** n_neighbors=5
- **Strengths:**
  - Simple and intuitive
  - No training required
  - Naturally handles multi-class problems
- **Weaknesses:**
  - Computationally expensive for large datasets
  - Sensitive to feature scaling
  - Requires significant memory

### Per-Class Performance

#### Decision Tree - Per-Class Metrics

**setosa:**
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**versicolor:**
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**virginica:**
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

#### KNN - Per-Class Metrics

**setosa:**
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**versicolor:**
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**virginica:**
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

### Business Implications

1. **Model Selection:** The analysis helps choose the most appropriate classifier for the specific use case.

2. **Performance Optimization:** Understanding strengths and weaknesses enables better model tuning.

3. **Interpretability vs. Accuracy:** Decision Tree provides interpretable rules while KNN may offer better accuracy in some cases.

4. **Scalability Considerations:** Decision Tree scales better for large datasets compared to KNN.

### Recommendations

1. **For Production Use:** Consider the {winner} based on this analysis
2. **Model Tuning:** Experiment with hyperparameters to improve performance
3. **Feature Engineering:** Explore additional features or transformations
4. **Cross-Validation:** Use k-fold cross-validation for more robust evaluation
5. **Ensemble Methods:** Consider combining both classifiers for improved performance

### Technical Notes

- **Random State:** Fixed for reproducibility
- **Stratified Split:** Ensures balanced class distribution in train/test sets
- **Metrics:** Macro-averaged for multi-class classification
- **Visualizations:** All charts saved in images/ directory

---
*Report generated automatically by Iris Classifier*
