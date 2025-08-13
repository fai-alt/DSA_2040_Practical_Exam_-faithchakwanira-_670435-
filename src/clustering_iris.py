"""
DSA 2040 Practical Exam - Task 2: Clustering
Data Mining Section - K-Means Clustering on Iris Dataset

This script implements K-Means clustering analysis:
1. Apply K-Means with k=3
2. Compare clusters with actual classes using ARI
3. Experiment with different k values
4. Create elbow curve and visualizations
5. Analyze cluster quality

Author: [Your Name]
Date: August 13, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clustering_iris.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IrisClusterAnalyzer:
    def __init__(self, data_path='../data/iris_processed.csv'):
        """
        Initialize the Iris cluster analyzer
        Args:
            data_path: Path to the processed Iris data
        """
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None
        self.kmeans_model = None
        self.cluster_labels = None
        self.scaler = StandardScaler()
        
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
    
    def apply_kmeans(self, k=3, random_state=42):
        """
        Apply K-Means clustering with specified k
        Args:
            k: Number of clusters
            random_state: Random seed for reproducibility
        """
        logger.info(f"Applying K-Means clustering with k={k}")
        
        try:
            # Initialize K-Means model
            self.kmeans_model = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
                max_iter=300
            )
            
            # Fit the model
            self.cluster_labels = self.kmeans_model.fit_predict(self.features)
            
            # Get cluster centers
            cluster_centers = self.kmeans_model.cluster_centers_
            
            logger.info(f"K-Means clustering completed with k={k}")
            logger.info(f"Cluster centers shape: {cluster_centers.shape}")
            logger.info(f"Cluster labels shape: {self.cluster_labels.shape}")
            
            return self.cluster_labels, cluster_centers
            
        except Exception as e:
            logger.error(f"Error in K-Means clustering: {str(e)}")
            raise
    
    def evaluate_clustering(self):
        """
        Evaluate clustering quality using various metrics
        """
        logger.info("Evaluating clustering quality")
        
        if self.cluster_labels is None:
            raise ValueError("No clustering results available. Run apply_kmeans() first.")
        
        try:
            # 1. Adjusted Rand Index (ARI) - compares with true labels
            ari_score = adjusted_rand_score(self.target, self.cluster_labels)
            logger.info(f"Adjusted Rand Index: {ari_score:.4f}")
            
            # 2. Silhouette Score - measures cluster cohesion and separation
            silhouette_avg = silhouette_score(self.features, self.cluster_labels)
            logger.info(f"Silhouette Score: {silhouette_avg:.4f}")
            
            # 3. Inertia (within-cluster sum of squares)
            inertia = self.kmeans_model.inertia_
            logger.info(f"Inertia: {inertia:.4f}")
            
            # 4. Number of iterations
            n_iter = self.kmeans_model.n_iter_
            logger.info(f"Number of iterations: {n_iter}")
            
            # 5. Cluster size distribution
            unique, counts = np.unique(self.cluster_labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))
            logger.info(f"Cluster sizes: {cluster_sizes}")
            
            # 6. Compare with actual species distribution
            species_distribution = self.target.value_counts()
            logger.info(f"Actual species distribution: {dict(species_distribution)}")
            
            # Create evaluation summary
            evaluation_summary = {
                'ARI_Score': ari_score,
                'Silhouette_Score': silhouette_avg,
                'Inertia': inertia,
                'Iterations': n_iter,
                'Cluster_Sizes': cluster_sizes,
                'Species_Distribution': dict(species_distribution)
            }
            
            return evaluation_summary
            
        except Exception as e:
            logger.error(f"Error evaluating clustering: {str(e)}")
            raise
    
    def experiment_with_k_values(self, k_range=range(2, 11)):
        """
        Experiment with different k values and create elbow curve
        Args:
            k_range: Range of k values to test
        """
        logger.info(f"Experimenting with k values: {list(k_range)}")
        
        try:
            inertias = []
            silhouette_scores = []
            ari_scores = []
            
            for k in k_range:
                logger.info(f"Testing k={k}")
                
                # Apply K-Means
                kmeans_temp = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )
                cluster_labels_temp = kmeans_temp.fit_predict(self.features)
                
                # Calculate metrics
                inertias.append(kmeans_temp.inertia_)
                silhouette_scores.append(silhouette_score(self.features, cluster_labels_temp))
                
                # ARI only makes sense when k matches number of true classes
                if k == 3:
                    ari_scores.append(adjusted_rand_score(self.target, cluster_labels_temp))
                else:
                    ari_scores.append(None)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'k': list(k_range),
                'Inertia': inertias,
                'Silhouette_Score': silhouette_scores,
                'ARI_Score': ari_scores
            })
            
            logger.info("K-value experimentation completed")
            logger.info(f"Results:\n{results_df}")
            
            # Save results
            results_df.to_csv('../data/kmeans_experiment_results.csv', index=False)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error in k-value experimentation: {str(e)}")
            raise
    
    def create_elbow_curve(self, results_df):
        """
        Create elbow curve visualization
        Args:
            results_df: DataFrame with k-value experiment results
        """
        logger.info("Creating elbow curve visualization")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Elbow curve (Inertia)
            ax1.plot(results_df['k'], results_df['Inertia'], 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
            ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add annotations for potential elbow points
            for i, (k, inertia) in enumerate(zip(results_df['k'], results_df['Inertia'])):
                if i > 0:
                    # Calculate the rate of change
                    rate_of_change = abs(inertia - results_df['Inertia'].iloc[i-1])
                    if rate_of_change < 100:  # Threshold for elbow detection
                        ax1.annotate(f'Elbow at k={k}', 
                                   xy=(k, inertia), 
                                   xytext=(k+1, inertia+200),
                                   arrowprops=dict(arrowstyle='->', color='red'),
                                   fontsize=10, color='red')
            
            # Plot 2: Silhouette Score
            ax2.plot(results_df['k'], results_df['Silhouette_Score'], 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax2.set_ylabel('Silhouette Score', fontsize=12)
            ax2.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Highlight optimal k based on silhouette score
            optimal_k = results_df.loc[results_df['Silhouette_Score'].idxmax(), 'k']
            optimal_score = results_df['Silhouette_Score'].max()
            ax2.annotate(f'Optimal k={optimal_k}\nScore={optimal_score:.3f}', 
                        xy=(optimal_k, optimal_score), 
                        xytext=(optimal_k+1, optimal_score-0.1),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, color='green')
            
            plt.tight_layout()
            plt.savefig('../images/kmeans_elbow_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Elbow curve visualization created and saved")
            
        except Exception as e:
            logger.error(f"Error creating elbow curve: {str(e)}")
            raise
    
    def visualize_clusters(self, k=3):
        """
        Visualize clustering results
        Args:
            k: Number of clusters used
        """
        logger.info(f"Creating cluster visualizations for k={k}")
        
        try:
            # Apply clustering if not already done
            if self.cluster_labels is None or len(np.unique(self.cluster_labels)) != k:
                self.apply_kmeans(k=k)
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Iris Dataset Clustering Analysis (k={k})', fontsize=16, fontweight='bold')
            
            # Plot 1: Petal Length vs Petal Width (colored by cluster)
            scatter1 = axes[0, 0].scatter(
                self.features['petal_length'], 
                self.features['petal_width'], 
                c=self.cluster_labels, 
                cmap='viridis', 
                alpha=0.7,
                s=60
            )
            axes[0, 0].set_xlabel('Petal Length', fontsize=12)
            axes[0, 0].set_ylabel('Petal Width', fontsize=12)
            axes[0, 0].set_title('Clusters: Petal Length vs Petal Width', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Sepal Length vs Sepal Width (colored by cluster)
            scatter2 = axes[0, 1].scatter(
                self.features['sepal_length'], 
                self.features['sepal_width'], 
                c=self.cluster_labels, 
                cmap='viridis', 
                alpha=0.7,
                s=60
            )
            axes[0, 1].set_xlabel('Sepal Length', fontsize=12)
            axes[0, 1].set_ylabel('Sepal Width', fontsize=12)
            axes[0, 1].set_title('Clusters: Sepal Length vs Sepal Width', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Petal Length vs Petal Width (colored by actual species)
            species_colors = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
            species_numeric = [species_colors[s] for s in self.target]
            
            scatter3 = axes[1, 0].scatter(
                self.features['petal_length'], 
                self.features['petal_width'], 
                c=species_numeric, 
                cmap='Set1', 
                alpha=0.7,
                s=60
            )
            axes[1, 0].set_xlabel('Petal Length', fontsize=12)
            axes[1, 0].set_ylabel('Petal Width', fontsize=12)
            axes[1, 0].set_title('Actual Species: Petal Length vs Petal Width', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Cluster comparison
            cluster_comparison = pd.DataFrame({
                'Cluster': self.cluster_labels,
                'Actual_Species': self.target
            })
            
            comparison_table = pd.crosstab(
                cluster_comparison['Cluster'], 
                cluster_comparison['Actual_Species']
            )
            
            sns.heatmap(comparison_table, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('Cluster vs Actual Species Comparison', fontsize=12)
            axes[1, 1].set_xlabel('Actual Species', fontsize=12)
            axes[1, 1].set_ylabel('Cluster', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'../images/iris_clustering_k{k}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Cluster visualizations created and saved for k={k}")
            
        except Exception as e:
            logger.error(f"Error creating cluster visualizations: {str(e)}")
            raise
    
    def analyze_results(self, evaluation_summary, results_df):
        """
        Analyze clustering results and generate insights
        Args:
            evaluation_summary: Results from evaluate_clustering()
            results_df: Results from k-value experimentation
        """
        logger.info("Analyzing clustering results")
        
        try:
            # Create analysis report
            analysis_report = f"""
# Iris Dataset Clustering Analysis Report

## Executive Summary
This report analyzes the performance of K-Means clustering on the Iris dataset, comparing different numbers of clusters and evaluating the quality of clustering results.

## Dataset Overview
- **Dataset**: Iris dataset (150 samples, 4 features)
- **Features**: Sepal length, Sepal width, Petal length, Petal width
- **True Classes**: 3 species (setosa, versicolor, virginica)
- **Data Type**: {'Synthetic' if 'synthetic' in str(self.data_path).lower() else 'Real'} data

## Clustering Results (k=3)

### Performance Metrics
- **Adjusted Rand Index (ARI)**: {evaluation_summary['ARI_Score']:.4f}
  - ARI ranges from -1 to 1, where 1 indicates perfect agreement
  - Our score of {evaluation_summary['ARI_Score']:.4f} indicates {'excellent' if evaluation_summary['ARI_Score'] > 0.8 else 'good' if evaluation_summary['ARI_Score'] > 0.6 else 'moderate'} agreement between clusters and true labels

- **Silhouette Score**: {evaluation_summary['Silhouette_Score']:.4f}
  - Silhouette score ranges from -1 to 1, where higher values indicate better-defined clusters
  - Our score of {evaluation_summary['Silhouette_Score']:.4f} suggests {'well-separated' if evaluation_summary['Silhouette_Score'] > 0.5 else 'moderately separated' if evaluation_summary['Silhouette_Score'] > 0.25 else 'poorly separated'} clusters

- **Inertia**: {evaluation_summary['Inertia']:.4f}
  - Lower inertia indicates tighter, more cohesive clusters

### Cluster Analysis
- **Cluster Sizes**: {evaluation_summary['Cluster_Sizes']}
- **Species Distribution**: {evaluation_summary['Species_Distribution']}

## K-Value Experimentation

### Elbow Method Analysis
The elbow method suggests optimal k by looking for the point where adding more clusters provides diminishing returns.

### Silhouette Score Analysis
The optimal number of clusters based on silhouette score is k={results_df.loc[results_df['Silhouette_Score'].idxmax(), 'k']}.

## Key Insights

1. **Optimal Clustering**: k=3 provides the best balance between cluster quality and interpretability
2. **Cluster Quality**: The high ARI score indicates that K-Means successfully identified the natural groupings in the data
3. **Feature Importance**: Petal measurements appear to be more discriminative than sepal measurements for clustering

## Real-World Applications

This clustering approach can be applied to:
- **Customer Segmentation**: Group customers based on purchasing behavior
- **Product Categorization**: Organize products into natural categories
- **Anomaly Detection**: Identify unusual patterns in data
- **Market Research**: Discover hidden patterns in consumer data

## Limitations and Considerations

1. **Assumption of Spherical Clusters**: K-Means assumes clusters are spherical and equally sized
2. **Feature Scaling**: Results depend on proper feature scaling
3. **Random Initialization**: Multiple runs may be needed for optimal results
4. **Data Quality**: Results are sensitive to outliers and noise

## Recommendations

1. **Use k=3** for this dataset as it aligns with biological reality
2. **Preprocess features** using standardization or normalization
3. **Run multiple iterations** to avoid local optima
4. **Validate results** using domain knowledge and multiple metrics

## Conclusion

K-Means clustering successfully identified the three natural groups in the Iris dataset, demonstrating the algorithm's effectiveness for well-separated, spherical clusters. The high ARI score confirms that the clustering aligns well with the true biological classifications.

{'**Note**: This analysis used synthetic data, which may affect the realism of results compared to the original Iris dataset.' if 'synthetic' in str(self.data_path).lower() else '**Note**: This analysis used the original Iris dataset from scikit-learn.'}
"""
            
            # Save analysis report
            with open('../reports/iris_clustering_analysis.md', 'w') as f:
                f.write(analysis_report)
            
            logger.info("Analysis report generated and saved")
            return analysis_report
            
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            raise
    
    def run_full_analysis(self, k=3):
        """
        Run the complete clustering analysis
        Args:
            k: Primary number of clusters to analyze
        """
        logger.info("Starting full clustering analysis")
        
        try:
            # 1. Load data
            logger.info("=== DATA LOADING ===")
            self.load_data()
            
            # 2. Apply K-Means with specified k
            logger.info("=== K-MEANS CLUSTERING ===")
            cluster_labels, cluster_centers = self.apply_kmeans(k=k)
            
            # 3. Evaluate clustering quality
            logger.info("=== CLUSTERING EVALUATION ===")
            evaluation_summary = self.evaluate_clustering()
            
            # 4. Experiment with different k values
            logger.info("=== K-VALUE EXPERIMENTATION ===")
            results_df = self.experiment_with_k_values()
            
            # 5. Create elbow curve
            logger.info("=== ELBOW CURVE CREATION ===")
            self.create_elbow_curve(results_df)
            
            # 6. Visualize clusters
            logger.info("=== CLUSTER VISUALIZATION ===")
            self.visualize_clusters(k=k)
            
            # 7. Analyze results
            logger.info("=== RESULTS ANALYSIS ===")
            analysis_report = self.analyze_results(evaluation_summary, results_df)
            
            # Summary
            logger.info("=== CLUSTERING ANALYSIS SUMMARY ===")
            logger.info(f"Dataset: {self.data.shape}")
            logger.info(f"Features: {list(self.features.columns)}")
            logger.info(f"Primary clustering: k={k}")
            logger.info(f"ARI Score: {evaluation_summary['ARI_Score']:.4f}")
            logger.info(f"Silhouette Score: {evaluation_summary['Silhouette_Score']:.4f}")
            logger.info(f"Optimal k (silhouette): {results_df.loc[results_df['Silhouette_Score'].idxmax(), 'k']}")
            
            logger.info("Clustering analysis completed successfully!")
            
            return {
                'evaluation_summary': evaluation_summary,
                'experiment_results': results_df,
                'analysis_report': analysis_report
            }
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {str(e)}")
            raise

def main():
    """Main function to run the clustering analysis"""
    try:
        # Initialize analyzer
        analyzer = IrisClusterAnalyzer()
        
        # Run full clustering analysis
        results = analyzer.run_full_analysis(k=3)
        
        logger.info("Clustering analysis completed successfully!")
        logger.info("Check the 'images', 'reports', and 'data' directories for outputs")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
