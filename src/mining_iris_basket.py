"""
DSA 2040 Practical Exam - Task 3: Classification and Association Rule Mining
Data Mining Section - Association Rule Mining

This script implements association rule mining:
1. Load synthetic transactional data
2. Apply Apriori algorithm
3. Generate association rules
4. Analyze rule implications

Author: [Your Name]
Date: August 13, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mining_iris_basket.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class AssociationRuleMiner:
    def __init__(self, data_path='../data/synthetic_transactions.csv'):
        """
        Initialize the association rule miner
        Args:
            data_path: Path to the transactional data
        """
        self.data_path = data_path
        self.transactions = None
        self.encoded_data = None
        self.frequent_itemsets = None
        self.rules = None
        
        # Create output directory if it doesn't exist
        os.makedirs('../images', exist_ok=True)
        os.makedirs('../reports', exist_ok=True)
        
    def load_transactional_data(self):
        """
        Load synthetic transactional data
        """
        try:
            logger.info(f"Loading transactional data from {self.data_path}")
            
            if os.path.exists(self.data_path):
                self.transactions = pd.read_csv(self.data_path)
                logger.info(f"Data loaded successfully: {self.transactions.shape}")
            else:
                logger.warning(f"Data file not found: {self.data_path}")
                logger.info("Generating synthetic transactional data...")
                self.generate_synthetic_transactions()
            
            return self.transactions
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def generate_synthetic_transactions(self):
        """
        Generate synthetic transactional data if file doesn't exist
        """
        logger.info("Generating synthetic transactional data")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define item pool (10 items - smaller for faster processing)
        items = [
            'Milk', 'Bread', 'Eggs', 'Butter', 'Cheese',
            'Yogurt', 'Bananas', 'Apples', 'Chicken', 'Beef'
        ]
        
        # Generate 30 transactions - smaller dataset
        num_transactions = 30
        transactions = []
        
        for i in range(num_transactions):
            # Each transaction has 3-8 items
            num_items = np.random.randint(3, 9)
            transaction_items = np.random.choice(items, num_items, replace=False)
            transactions.append(list(transaction_items))
        
        self.transactions = pd.DataFrame({
            'TransactionID': range(1, num_transactions + 1),
            'Items': transactions
        })
        
        # Save to file
        self.transactions.to_csv(self.data_path, index=False)
        logger.info(f"Generated {len(transactions)} synthetic transactions")
        
        return self.transactions
    
    def prepare_data_for_mining(self):
        """
        Prepare transactional data for association rule mining
        """
        logger.info("Preparing data for association rule mining")
        
        try:
            # Extract items from transactions
            if 'Items' in self.transactions.columns:
                transaction_list = self.transactions['Items'].tolist()
            else:
                # If data is already in list format
                transaction_list = self.transactions.values.tolist()
            
            # Use TransactionEncoder to convert to one-hot encoding
            te = TransactionEncoder()
            te_ary = te.fit(transaction_list).transform(transaction_list)
            self.encoded_data = pd.DataFrame(te_ary, columns=te.columns_)
            
            logger.info(f"Data encoded successfully: {self.encoded_data.shape}")
            logger.info(f"Number of unique items: {len(te.columns_)}")
            
            return self.encoded_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def apply_apriori_algorithm(self, min_support=0.2):
        """
        Apply Apriori algorithm to find frequent itemsets
        Args:
            min_support: Minimum support threshold
        """
        logger.info(f"Applying Apriori algorithm with min_support={min_support}")
        
        try:
            # Find frequent itemsets
            self.frequent_itemsets = apriori(self.encoded_data, 
                                           min_support=min_support, 
                                           use_colnames=True)
            
            logger.info(f"Found {len(self.frequent_itemsets)} frequent itemsets")
            
            # Sort by support
            self.frequent_itemsets = self.frequent_itemsets.sort_values('support', ascending=False)
            
            return self.frequent_itemsets
            
        except Exception as e:
            logger.error(f"Error applying Apriori algorithm: {str(e)}")
            raise
    
    def generate_association_rules(self, min_confidence=0.5):
        """
        Generate association rules from frequent itemsets
        Args:
            min_confidence: Minimum confidence threshold
        """
        logger.info(f"Generating association rules with min_confidence={min_confidence}")
        
        try:
            # Generate rules
            self.rules = association_rules(self.frequent_itemsets, 
                                        metric="confidence", 
                                        min_threshold=min_confidence)
            
            # Calculate lift
            self.rules['lift'] = self.rules['lift'].round(3)
            
            # Sort by lift (descending)
            self.rules = self.rules.sort_values('lift', ascending=False)
            
            logger.info(f"Generated {len(self.rules)} association rules")
            
            return self.rules
            
        except Exception as e:
            logger.error(f"Error generating association rules: {str(e)}")
            raise
    
    def display_top_rules(self, top_n=10):
        """
        Display top association rules
        Args:
            top_n: Number of top rules to display
        """
        if self.rules is None or self.rules.empty:
            logger.warning("No rules available to display")
            return
        
        logger.info(f"Displaying top {top_n} association rules")
        
        # Select top rules
        top_rules = self.rules.head(top_n)
        
        print(f"\n{'='*80}")
        print(f"TOP {top_n} ASSOCIATION RULES")
        print(f"{'='*80}")
        print(f"{'Antecedent':<30} {'Consequent':<30} {'Support':<10} {'Confidence':<12} {'Lift':<8}")
        print(f"{'-'*30} {'-'*30} {'-'*10} {'-'*12} {'-'*8}")
        
        for idx, rule in top_rules.iterrows():
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))
            support = f"{rule['support']:.3f}"
            confidence = f"{rule['confidence']:.3f}"
            lift = f"{rule['lift']:.3f}"
            
            print(f"{antecedent:<30} {consequent:<30} {support:<10} {confidence:<12} {lift:<8}")
        
        print(f"{'='*80}")
        
        return top_rules
    
    def create_support_distribution_chart(self):
        """Create chart showing support distribution of frequent itemsets"""
        logger.info("Creating support distribution chart")
        
        if self.frequent_itemsets is None or self.frequent_itemsets.empty:
            logger.error("No frequent itemsets available for visualization")
            return False
        
        plt.figure(figsize=(12, 8))
        
        # Plot support distribution
        plt.hist(self.frequent_itemsets['support'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.2, color='red', linestyle='--', label='Min Support Threshold (0.2)')
        
        plt.title('Distribution of Support Values for Frequent Itemsets', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Support', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the chart
        output_path = '../images/support_distribution_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Support distribution chart saved to: {output_path}")
        
        plt.show()
        return True
    
    def create_lift_confidence_scatter(self):
        """Create scatter plot of lift vs confidence"""
        logger.info("Creating lift vs confidence scatter plot")
        
        if self.rules is None or self.rules.empty:
            logger.error("No rules available for visualization")
            return False
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        plt.scatter(self.rules['confidence'], self.rules['lift'], 
                   alpha=0.6, s=50, c=self.rules['support'], cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Support', fontsize=12)
        
        # Add threshold lines
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Lift = 1 (Independent)')
        plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Min Confidence = 0.5')
        
        plt.title('Association Rules: Lift vs Confidence', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Lift', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the chart
        output_path = '../images/lift_confidence_scatter.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Lift vs confidence scatter plot saved to: {output_path}")
        
        plt.show()
        return True
    
    def create_top_rules_chart(self, top_n=10):
        """Create bar chart of top rules by lift"""
        logger.info(f"Creating top {top_n} rules chart")
        
        if self.rules is None or self.rules.empty:
            logger.error("No rules available for visualization")
            return False
        
        # Select top rules
        top_rules = self.rules.head(top_n)
        
        plt.figure(figsize=(14, 10))
        
        # Create bar chart
        rule_labels = []
        for _, rule in top_rules.iterrows():
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))
            rule_labels.append(f"{antecedent} -> {consequent}")
        
        bars = plt.barh(range(len(top_rules)), top_rules['lift'], 
                       color=sns.color_palette("husl", len(top_rules)))
        
        # Customize the chart
        plt.title(f'Top {top_n} Association Rules by Lift', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Lift', fontsize=12)
        plt.ylabel('Rules', fontsize=12)
        plt.yticks(range(len(top_rules)), rule_labels, fontsize=10)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_rules['lift'])):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        # Add grid
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        output_path = '../images/top_rules_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Top rules chart saved to: {output_path}")
        
        plt.show()
        return True
    
    def analyze_rule_implications(self):
        """Analyze implications of top association rules"""
        logger.info("Analyzing rule implications")
        
        if self.rules is None or self.rules.empty:
            logger.warning("No rules available for analysis")
            return None
        
        # Get top 5 rules
        top_rules = self.rules.head(5)
        
        analysis = """
# Association Rule Mining Analysis Report
## DSA 2040 Practical Exam - Task 3 Part B

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Executive Summary
This report analyzes the implications of association rules discovered in retail transactional data using the Apriori algorithm.

### Top Association Rules Analysis

"""
        
        for i, (idx, rule) in enumerate(top_rules.iterrows(), 1):
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))
            support = rule['support']
            confidence = rule['confidence']
            lift = rule['lift']
            
            analysis += f"""
            #### Rule {i}: {antecedent} -> {consequent}

- **Support:** {support:.3f} ({support*100:.1f}% of transactions contain both items)
- **Confidence:** {confidence:.3f} ({confidence*100:.1f}% of transactions with {antecedent} also contain {consequent})
- **Lift:** {lift:.3f} (Rule is {lift:.1f}x more likely than random chance)

**Business Implications:**
"""
            
            if lift > 1.5:
                analysis += f"- Strong positive association between {antecedent} and {consequent}\n"
                analysis += f"- Consider cross-selling strategies and product placement\n"
                analysis += f"- Bundle these items together in promotions\n"
            elif lift > 1.0:
                analysis += f"- Moderate positive association between {antecedent} and {consequent}\n"
                analysis += f"- Potential for targeted marketing campaigns\n"
            else:
                analysis += f"- Weak association between {antecedent} and {consequent}\n"
                analysis += f"- Consider if these items should be separated\n"
            
            analysis += f"- **Recommendation:** Use this insight for inventory planning and marketing strategies\n\n"
        
        analysis += """
### Overall Business Insights

1. **Cross-Selling Opportunities:** Rules with high lift indicate strong product associations that can be leveraged for cross-selling.

2. **Inventory Management:** Understanding product relationships helps optimize inventory levels and product placement.

3. **Marketing Strategies:** Use association rules to design targeted promotions and bundle offers.

4. **Store Layout:** Consider placing strongly associated products near each other to increase sales.

### Technical Notes

- **Min Support:** 0.5 (50% of transactions must contain the itemset)
- **Min Confidence:** 0.8 (80% confidence threshold for rule generation)
- **Lift Interpretation:** 
  - Lift > 1: Positive association
  - Lift = 1: Independent items
  - Lift < 1: Negative association

### Recommendations

1. **High-Lift Rules:** Focus marketing efforts on products with strong associations
2. **Support Analysis:** Consider rules with reasonable support for practical implementation
3. **Confidence Threshold:** Adjust confidence based on business requirements
4. **Regular Updates:** Re-run analysis periodically as customer behavior changes

---
*Report generated automatically by Association Rule Miner*
"""
        
        # Save analysis report
        output_path = '../reports/association_rules_analysis.md'
        with open(output_path, 'w') as f:
            f.write(analysis)
        
        logger.info(f"Analysis report saved to: {output_path}")
        return analysis
    
    def run_full_analysis(self, min_support=0.5, min_confidence=0.8):
        """
        Run complete association rule mining analysis
        Args:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
        """
        logger.info("Starting complete association rule mining analysis")
        
        try:
            # Load data
            self.load_transactional_data()
            
            # Prepare data
            self.prepare_data_for_mining()
            
            # Apply Apriori algorithm
            self.apply_apriori_algorithm(min_support)
            
            # Generate rules
            self.generate_association_rules(min_confidence)
            
            # Display results
            self.display_top_rules(10)
            
            # Create visualizations
            self.create_support_distribution_chart()
            self.create_lift_confidence_scatter()
            self.create_top_rules_chart(10)
            
            # Generate analysis report
            self.analyze_rule_implications()
            
            logger.info("Association rule mining analysis completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return False

def main():
    """Main function to run association rule mining"""
    try:
        # Initialize miner
        miner = AssociationRuleMiner()
        
        # Run full analysis with very high thresholds for instant execution
        success = miner.run_full_analysis(min_support=0.5, min_confidence=0.8)
        
        if success:
            logger.info("Association rule mining completed successfully!")
            print("✅ Association rule mining completed successfully!")
            print("📊 Check the images/ directory for charts")
            print("📋 Check the reports/ directory for analysis report")
        else:
            logger.warning("Association rule mining failed")
            print("⚠️ Association rule mining failed. Check logs for details.")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
