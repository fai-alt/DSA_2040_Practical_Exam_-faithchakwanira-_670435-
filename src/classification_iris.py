"""
DSA 2040 Practical Exam - Task 3: Classification and Association Rule Mining
Data Mining Section - Classification (Part A)

This script implements classification analysis:
1. Train Decision Tree classifier
2. Compare with KNN classifier
3. Compute performance metrics
4. Visualize decision tree

Author: [Your Name]
Date: August 13, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification_iris.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class IrisClassifier:
    def __init__(self, data_path='../data/iris_processed.csv'):
        """
        Initialize the Iris classifier
        Args:
            data_path: Path to the processed Iris data
        """
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.decision_tree = None
        self.knn = None
        self.dt_predictions = None
        self.knn_predictions = None
        
        # Create output directory if it doesn't exist
        os.makedirs('../images', exist_ok=True)
        os.makedirs('../reports', exist_ok=True)
        
    def load_data(self):
        """
        Load the processed Iris data
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            if os.path.exists(self.data_path):
                self.data = pd.read_csv(self.data_path)
                logger.info(f"Data loaded successfully: {self.data.shape}")
            else:
                logger.warning(f"Data file not found: {self.data_path}")
                logger.info("Attempting to load from train data...")
                train_path = '../data/iris_train.csv'
                if os.path.exists(train_path):
                    self.data = pd.read_csv(train_path)
                    logger.info(f"Train data loaded: {self.data.shape}")
                else:
                    raise FileNotFoundError("No data files found. Run preprocessing first.")
            
            # Separate features and target
            self.features = self.data.drop('species', axis=1)
            self.target = self.data['species']
            
            logger.info(f"Features shape: {self.features.shape}")
            logger.info(f"Target classes: {list(self.target.unique())}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        logger.info(f"Splitting data with test_size={test_size}")
        
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.features, self.target, test_size=test_size, random_state=random_state, stratify=self.target
            )
            
            logger.info(f"Training set: {self.X_train.shape}")
            logger.info(f"Testing set: {self.X_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def train_decision_tree(self, random_state=42):
        """
        Train Decision Tree classifier
        Args:
            random_state: Random seed for reproducibility
        """
        logger.info("Training Decision Tree classifier")
        
        try:
            self.decision_tree = DecisionTreeClassifier(
                random_state=random_state,
                max_depth=5,  # Limit depth to prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            self.decision_tree.fit(self.X_train, self.y_train)
            
            # Make predictions
            self.dt_predictions = self.decision_tree.predict(self.X_test)
            
            logger.info("Decision Tree trained successfully")
            return self.decision_tree
            
        except Exception as e:
            logger.error(f"Error training Decision Tree: {str(e)}")
            raise
    
    def train_knn(self, n_neighbors=5):
        """
        Train K-Nearest Neighbors classifier
        Args:
            n_neighbors: Number of neighbors to consider
        """
        logger.info(f"Training KNN classifier with k={n_neighbors}")
        
        try:
            self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.knn.fit(self.X_train, self.y_train)
            
            # Make predictions
            self.knn_predictions = self.knn.predict(self.X_test)
            
            logger.info("KNN classifier trained successfully")
            return self.knn
            
        except Exception as e:
            logger.error(f"Error training KNN: {str(e)}")
            raise
    
    def compute_metrics(self, y_true, y_pred, classifier_name):
        """
        Compute classification metrics
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classifier_name: Name of the classifier for logging
        """
        logger.info(f"Computing metrics for {classifier_name}")
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro'),
                'recall_macro': recall_score(y_true, y_pred, average='macro'),
                'f1_macro': f1_score(y_true, y_pred, average='macro')
            }
            
            # Compute per-class metrics
            precision_per_class = precision_score(y_true, y_pred, average=None)
            recall_per_class = recall_score(y_true, y_pred, average=None)
            f1_per_class = f1_score(y_true, y_pred, average=None)
            
            logger.info(f"{classifier_name} Metrics:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
            logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
            logger.info(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
            
            return metrics, precision_per_class, recall_per_class, f1_per_class
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            raise
    
    def compare_classifiers(self):
        """
        Compare Decision Tree and KNN classifiers
        """
        logger.info("Comparing Decision Tree and KNN classifiers")
        
        try:
            # Compute metrics for both classifiers
            dt_metrics, dt_precision, dt_recall, dt_f1 = self.compute_metrics(
                self.y_test, self.dt_predictions, "Decision Tree"
            )
            
            knn_metrics, knn_precision, knn_recall, knn_f1 = self.compute_metrics(
                self.y_test, self.knn_predictions, "KNN"
            )
            
            # Create comparison DataFrame
            comparison_data = {
                'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
                'Decision Tree': [
                    dt_metrics['accuracy'],
                    dt_metrics['precision_macro'],
                    dt_metrics['recall_macro'],
                    dt_metrics['f1_macro']
                ],
                'KNN (k=5)': [
                    knn_metrics['accuracy'],
                    knn_metrics['precision_macro'],
                    knn_metrics['recall_macro'],
                    knn_metrics['f1_macro']
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Determine which classifier is better
            dt_better = 0
            knn_better = 0
            
            for i in range(1, 4):  # Skip accuracy, compare other metrics
                if comparison_df.iloc[i, 1] > comparison_df.iloc[i, 2]:
                    dt_better += 1
                elif comparison_df.iloc[i, 1] < comparison_df.iloc[i, 2]:
                    knn_better += 1
            
            # Overall winner
            if dt_better > knn_better:
                winner = "Decision Tree"
                reason = "Decision Tree performs better on most metrics"
            elif knn_better > dt_better:
                winner = "KNN"
                reason = "KNN performs better on most metrics"
            else:
                winner = "Tie"
                reason = "Both classifiers perform similarly"
            
            logger.info(f"Classifier Comparison Results:")
            logger.info(f"  Decision Tree better on: {dt_better} metrics")
            logger.info(f"  KNN better on: {knn_better} metrics")
            logger.info(f"  Winner: {winner}")
            logger.info(f"  Reason: {reason}")
            
            return comparison_df, winner, reason
            
        except Exception as e:
            logger.error(f"Error comparing classifiers: {str(e)}")
            raise
    
    def create_confusion_matrices(self):
        """Create confusion matrices for both classifiers"""
        logger.info("Creating confusion matrices")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Decision Tree confusion matrix
            cm_dt = confusion_matrix(self.y_test, self.dt_predictions)
            sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Decision Tree Confusion Matrix', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # KNN confusion matrix
            cm_knn = confusion_matrix(self.y_test, self.knn_predictions)
            sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=ax2)
            ax2.set_title('KNN Confusion Matrix', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            
            plt.tight_layout()
            
            # Save the chart
            output_path = '../images/confusion_matrices.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to: {output_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            logger.error(f"Error creating confusion matrices: {str(e)}")
            return False
    
    def visualize_decision_tree(self):
        """Visualize the decision tree"""
        logger.info("Visualizing decision tree")
        
        try:
            plt.figure(figsize=(20, 12))
            
            # Plot the decision tree
            plot_tree(self.decision_tree, 
                     feature_names=self.features.columns,
                     class_names=self.target.unique(),
                     filled=True,
                     rounded=True,
                     fontsize=10)
            
            plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold', pad=20)
            
            # Save the chart
            output_path = '../images/decision_tree_visualization.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Decision tree visualization saved to: {output_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing decision tree: {str(e)}")
            return False
    
    def create_metrics_comparison_chart(self, comparison_df):
        """Create bar chart comparing metrics between classifiers"""
        logger.info("Creating metrics comparison chart")
        
        try:
            # Prepare data for plotting
            metrics = comparison_df['Metric'].values
            dt_scores = comparison_df['Decision Tree'].values
            knn_scores = comparison_df['KNN (k=5)'].values
            
            x = np.arange(len(metrics))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create bars
            bars1 = ax.bar(x - width/2, dt_scores, width, label='Decision Tree', color='skyblue', alpha=0.8)
            bars2 = ax.bar(x + width/2, knn_scores, width, label='KNN (k=5)', color='lightcoral', alpha=0.8)
            
            # Customize the chart
            ax.set_title('Classifier Performance Comparison', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save the chart
            output_path = '../images/metrics_comparison_chart.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison chart saved to: {output_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            logger.error(f"Error creating metrics comparison chart: {str(e)}")
            return False
    
    def generate_classification_report(self, comparison_df, winner, reason):
        """Generate comprehensive classification analysis report"""
        logger.info("Generating classification analysis report")
        
        try:
            # Get detailed classification reports
            dt_report = classification_report(self.y_test, self.dt_predictions, output_dict=True)
            knn_report = classification_report(self.y_test, self.knn_predictions, output_dict=True)
            
            report = f"""
# Classification Analysis Report
## DSA 2040 Practical Exam - Task 3 Part A

**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

### Executive Summary
This report analyzes the performance of Decision Tree and K-Nearest Neighbors (KNN) classifiers on the Iris dataset.

### Dataset Information
- **Dataset:** Iris dataset (processed)
- **Features:** 4 numerical features (sepal length/width, petal length/width)
- **Classes:** 3 species (setosa, versicolor, virginica)
- **Training Set:** {len(self.X_train)} samples
- **Testing Set:** {len(self.X_test)} samples

### Classifier Performance Comparison

| Metric | Decision Tree | KNN (k=5) |
|--------|---------------|------------|
"""
            
            for i, metric in enumerate(comparison_df['Metric']):
                dt_val = comparison_df.iloc[i, 1]
                knn_val = comparison_df.iloc[i, 2]
                report += f"| {metric} | {dt_val:.4f} | {knn_val:.4f} |\n"
            
            report += f"""

### Winner Analysis
**Best Classifier:** {winner}
**Reason:** {reason}

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
"""
            
            for class_name in self.target.unique():
                if class_name in dt_report:
                    class_metrics = dt_report[class_name]
                    report += f"""
**{class_name}:**
- Precision: {class_metrics['precision']:.4f}
- Recall: {class_metrics['recall']:.4f}
- F1-Score: {class_metrics['f1-score']:.4f}
"""
            
            report += """
#### KNN - Per-Class Metrics
"""
            
            for class_name in self.target.unique():
                if class_name in knn_report:
                    class_metrics = knn_report[class_name]
                    report += f"""
**{class_name}:**
- Precision: {class_metrics['precision']:.4f}
- Recall: {class_metrics['recall']:.4f}
- F1-Score: {class_metrics['f1-score']:.4f}
"""
            
            report += """
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
"""
            
            # Save report
            output_path = '../reports/classification_analysis_report.md'
            with open(output_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Classification report saved to: {output_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def run_full_classification(self):
        """Run complete classification analysis"""
        logger.info("Starting complete classification analysis")
        
        try:
            # Load and prepare data
            self.load_data()
            self.split_data()
            
            # Train classifiers
            self.train_decision_tree()
            self.train_knn()
            
            # Compare performance
            comparison_df, winner, reason = self.compare_classifiers()
            
            # Create visualizations
            self.create_confusion_matrices()
            self.visualize_decision_tree()
            self.create_metrics_comparison_chart(comparison_df)
            
            # Generate report
            self.generate_classification_report(comparison_df, winner, reason)
            
            logger.info("Classification analysis completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during classification analysis: {str(e)}")
            return False

def main():
    """Main function to run classification analysis"""
    try:
        # Initialize classifier
        classifier = IrisClassifier()
        
        # Run full analysis
        success = classifier.run_full_classification()
        
        if success:
            logger.info("Classification analysis completed successfully!")
            print("✅ Classification analysis completed successfully!")
            print("📊 Check the images/ directory for charts")
            print("📋 Check the reports/ directory for analysis report")
        else:
            logger.warning("Classification analysis failed")
            print("⚠️ Classification analysis failed. Check logs for details.")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
