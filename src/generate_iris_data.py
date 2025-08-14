#!/usr/bin/env python3
"""
Synthetic Iris Data Generator for DSA 2040 Practical Exam
Generates 150 samples with 3 clusters mimicking Iris species
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_iris_data(n_samples=150, n_clusters=3):
    """
    Generate synthetic Iris-like data with 3 clusters
    
    Args:
        n_samples (int): Number of samples to generate (default: 150)
        n_clusters (int): Number of clusters (default: 3)
    
    Returns:
        tuple: (features, labels, feature_names)
    """
    
    # Feature names matching Iris dataset
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Generate cluster centers that mimic Iris species characteristics
    # Cluster 0: Small flowers (like Iris setosa)
    # Cluster 1: Medium flowers (like Iris versicolor) 
    # Cluster 2: Large flowers (like Iris virginica)
    
    centers = np.array([
        [5.0, 3.5, 1.5, 0.2],  # Small flowers
        [6.0, 3.0, 4.5, 1.5],  # Medium flowers
        [6.5, 3.0, 5.5, 2.0]   # Large flowers
    ])
    
    # Generate data using make_blobs with realistic standard deviations
    # Use a single std dev for all clusters (make_blobs expects one value or one per cluster)
    cluster_std = 0.4  # Realistic variation for all clusters
    
    # Generate features
    features, labels = make_blobs(
        n_samples=n_samples,
        n_features=4,
        centers=centers,
        cluster_std=cluster_std,
        random_state=42
    )
    
    # Ensure realistic ranges (similar to actual Iris dataset)
    # Sepal length: 4.3 - 7.9
    features[:, 0] = np.clip(features[:, 0], 4.0, 8.0)
    # Sepal width: 2.0 - 4.4
    features[:, 1] = np.clip(features[:, 1], 2.0, 4.5)
    # Petal length: 1.0 - 6.9
    features[:, 2] = np.clip(features[:, 2], 1.0, 7.0)
    # Petal width: 0.1 - 2.5
    features[:, 3] = np.clip(features[:, 3], 0.1, 2.5)
    
    # Round to 1 decimal place for realistic precision
    features = np.round(features, 1)
    
    return features, labels, feature_names

def create_iris_dataframe(features, labels, feature_names):
    """
    Create a pandas DataFrame with the synthetic Iris data
    
    Args:
        features (np.array): Feature values
        labels (np.array): Cluster labels
        feature_names (list): Names of features
    
    Returns:
        pd.DataFrame: DataFrame with features and target
    """
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    
    # Add target column with species names
    species_names = ['setosa', 'versicolor', 'virginica']
    df['species'] = [species_names[label] for label in labels]
    
    # Add target column with numeric labels (0, 1, 2)
    df['target'] = labels
    
    return df

def visualize_generated_data(df):
    """
    Create visualizations to verify the generated data
    
    Args:
        df (pd.DataFrame): DataFrame with features and target
    """
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Synthetic Iris Data Visualization', fontsize=16, fontweight='bold')
    
    # 1. Sepal length vs Sepal width
    axes[0, 0].scatter(df['sepal_length'], df['sepal_width'], 
                       c=df['target'], cmap='viridis', alpha=0.7, s=60)
    axes[0, 0].set_xlabel('Sepal Length (cm)')
    axes[0, 0].set_ylabel('Sepal Width (cm)')
    axes[0, 0].set_title('Sepal Length vs Sepal Width')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Petal length vs Petal width
    axes[0, 1].scatter(df['petal_length'], df['petal_width'], 
                       c=df['target'], cmap='viridis', alpha=0.7, s=60)
    axes[0, 1].set_xlabel('Petal Length (cm)')
    axes[0, 1].set_ylabel('Petal Width (cm)')
    axes[0, 1].set_title('Petal Length vs Petal Width')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sepal length vs Petal length
    axes[1, 0].scatter(df['sepal_length'], df['petal_length'], 
                       c=df['target'], cmap='viridis', alpha=0.7, s=60)
    axes[1, 0].set_xlabel('Sepal Length (cm)')
    axes[1, 0].set_ylabel('Petal Length (cm)')
    axes[1, 0].set_title('Sepal Length vs Petal Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot of features by species
    df_melted = df.melt(id_vars=['species'], 
                        value_vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                        var_name='Feature', value_name='Value')
    
    sns.boxplot(data=df_melted, x='Feature', y='Value', hue='species', ax=axes[1, 1])
    axes[1, 1].set_title('Feature Distribution by Species')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '../images/synthetic_iris_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    plt.show()

def main():
    """Main function to generate and save synthetic Iris data"""
    
    print("Generating synthetic Iris data for DSA 2040 Practical Exam...")
    
    # Generate data
    features, labels, feature_names = generate_synthetic_iris_data(150, 3)
    
    # Create DataFrame
    iris_df = create_iris_dataframe(features, labels, feature_names)
    
    # Display basic information
    print(f"\nGenerated {len(iris_df)} samples of synthetic Iris data")
    print(f"Features: {feature_names}")
    print(f"Target classes: {iris_df['species'].unique()}")
    print(f"Class distribution:")
    print(iris_df['species'].value_counts())
    
    # Display sample data
    print("\nSample data:")
    print(iris_df.head(10))
    
    # Display summary statistics
    print("\nSummary statistics by species:")
    print(iris_df.groupby('species').describe())
    
    # Save to CSV
    output_file = '../data/synthetic_iris_data.csv'
    iris_df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_generated_data(iris_df)
    
    # Display data quality checks
    print("\nData quality checks:")
    print(f"Missing values: {iris_df.isnull().sum().sum()}")
    print(f"Feature ranges:")
    for feature in feature_names:
        print(f"  {feature}: {iris_df[feature].min():.1f} - {iris_df[feature].max():.1f}")
    
    return iris_df

if __name__ == "__main__":
    iris_data = main()
