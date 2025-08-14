"""
DSA 2040 Practical Exam - Task 3: OLAP Queries and Analysis
Data Warehousing Section - OLAP Visualizations

This script creates visualizations for OLAP query results:
1. Bar chart of sales by country
2. Time series analysis
3. Customer segmentation charts
4. Product performance analysis

Author: [Your Name]
Date: August 13, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('olap_visualizations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class OLAPVisualizer:
    def __init__(self, db_path='../retail_dw.db'):
        """
        Initialize the OLAP visualizer
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.conn = None
        
        # Create output directory if it doesn't exist
        os.makedirs('../images', exist_ok=True)
        os.makedirs('../reports', exist_ok=True)
        
    def connect_database(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            return False
            
    def execute_query(self, query):
        """Execute a SQL query and return results as DataFrame"""
        try:
            if self.conn is None:
                if not self.connect_database():
                    return None
            
            df = pd.read_sql_query(query, self.conn)
            logger.info(f"Query executed successfully: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return None
    
    def create_sales_by_country_chart(self):
        """
        Create bar chart of sales by country
        This is the main visualization required for Task 3
        """
        logger.info("Creating sales by country chart")
        
        query = """
        SELECT 
            l.Country,
            SUM(f.TotalSales) as TotalSales,
            COUNT(f.SaleID) as TransactionCount
        FROM SalesFact f
        JOIN LocationDim l ON f.LocationID = l.LocationID
        GROUP BY l.Country
        ORDER BY TotalSales DESC
        """
        
        df = self.execute_query(query)
        if df is None or df.empty:
            logger.error("No data available for sales by country chart")
            return False
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        bars = plt.bar(range(len(df)), df['TotalSales'], 
                      color=sns.color_palette("husl", len(df)))
        
        # Customize the chart
        plt.title('Total Sales by Country', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.xticks(range(len(df)), df['Country'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df['TotalSales'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['TotalSales'])*0.01,
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        output_path = '../images/sales_by_country_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sales by country chart saved to: {output_path}")
        
        plt.show()
        return True
    
    def create_time_series_analysis(self):
        """Create time series analysis of sales trends"""
        logger.info("Creating time series analysis")
        
        query = """
        SELECT 
            t.Year,
            t.Month,
            t.MonthName,
            SUM(f.TotalSales) as TotalSales,
            COUNT(f.SaleID) as TransactionCount
        FROM SalesFact f
        JOIN TimeDim t ON f.TimeID = t.TimeID
        GROUP BY t.Year, t.Month, t.MonthName
        ORDER BY t.Year, t.Month
        """
        
        df = self.execute_query(query)
        if df is None or df.empty:
            logger.error("No data available for time series analysis")
            return False
            
        # Create the visualization
        plt.figure(figsize=(14, 8))
        
        # Create line plot
        plt.plot(range(len(df)), df['TotalSales'], marker='o', linewidth=2, markersize=6)
        
        # Customize the chart
        plt.title('Sales Trends Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        
        # Set x-axis labels
        x_labels = [f"{row['MonthName']} {row['Year']}" for _, row in df.iterrows()]
        plt.xticks(range(len(df)), x_labels, rotation=45, ha='right')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        output_path = '../images/sales_time_series.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Time series chart saved to: {output_path}")
        
        plt.show()
        return True
        
    def create_customer_segmentation_chart(self):
        """Create customer segmentation analysis chart"""
        logger.info("Creating customer segmentation chart")
        
        query = """
        SELECT 
            c.CustomerSegment,
            COUNT(DISTINCT f.CustomerID) as CustomerCount,
            SUM(f.TotalSales) as TotalSales
        FROM SalesFact f
        JOIN CustomerDim c ON f.CustomerID = c.CustomerID
        GROUP BY c.CustomerSegment
        ORDER BY TotalSales DESC
        """
        
        df = self.execute_query(query)
        if df is None or df.empty:
            logger.error("No data available for customer segmentation chart")
            return False
            
        # Create the visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Pie chart for customer count
        ax1.pie(df['CustomerCount'], labels=df['CustomerSegment'], autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette("husl", len(df)))
        ax1.set_title('Customer Distribution by Segment', fontsize=14, fontweight='bold')
        
        # Bar chart for total sales
        bars = ax2.bar(range(len(df)), df['TotalSales'], 
                       color=sns.color_palette("husl", len(df)))
        ax2.set_title('Total Sales by Customer Segment', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Customer Segment')
        ax2.set_ylabel('Total Sales ($)')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['CustomerSegment'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df['TotalSales'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['TotalSales'])*0.01,
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the chart
        output_path = '../images/customer_segmentation_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Customer segmentation chart saved to: {output_path}")
        
        plt.show()
        return True
        
    def create_product_performance_chart(self):
        """Create product performance analysis chart"""
        logger.info("Creating product performance chart")
        
        query = """
        SELECT 
            p.Category,
            SUM(f.TotalSales) as TotalSales,
            SUM(f.Quantity) as TotalQuantity,
            COUNT(f.SaleID) as TransactionCount
        FROM SalesFact f
        JOIN ProductDim p ON f.ProductID = p.ProductID
        GROUP BY p.Category
        ORDER BY TotalSales DESC
        """
        
        df = self.execute_query(query)
        if df is None or df.empty:
            logger.error("No data available for product performance chart")
            return False
            
        # Create the visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar chart for sales by category
        bars1 = ax1.bar(range(len(df)), df['TotalSales'], 
                        color=sns.color_palette("husl", len(df)))
        ax1.set_title('Total Sales by Product Category', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Product Category')
        ax1.set_ylabel('Total Sales ($)')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['Category'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars1, df['TotalSales'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['TotalSales'])*0.01,
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Bar chart for quantity sold by category
        bars2 = ax2.bar(range(len(df)), df['TotalQuantity'], 
                        color=sns.color_palette("husl", len(df)))
        ax2.set_title('Total Quantity Sold by Product Category', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Product Category')
        ax2.set_ylabel('Total Quantity')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['Category'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars2, df['TotalQuantity'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['TotalQuantity'])*0.01,
                    f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the chart
        output_path = '../images/product_performance_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Product performance chart saved to: {output_path}")
        
        plt.show()
        return True
        
    def create_geographic_analysis_chart(self):
        """Create geographic sales distribution chart"""
        logger.info("Creating geographic analysis chart")
        
        query = """
        SELECT 
            l.Country,
            l.Region,
            SUM(f.TotalSales) as TotalSales,
            COUNT(f.SaleID) as TransactionCount
        FROM SalesFact f
        JOIN LocationDim l ON f.LocationID = l.LocationID
        GROUP BY l.Country, l.Region
        ORDER BY TotalSales DESC
        """
        
        df = self.execute_query(query)
        if df is None or df.empty:
            logger.error("No data available for geographic analysis chart")
            return False
            
        # Create the visualization
        plt.figure(figsize=(14, 8))
        
        # Create horizontal bar chart for better country name readability
        bars = plt.barh(range(len(df)), df['TotalSales'], 
                       color=sns.color_palette("husl", len(df)))
        
        # Customize the chart
        plt.title('Geographic Sales Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Total Sales ($)', fontsize=12)
        plt.ylabel('Country', fontsize=12)
        plt.yticks(range(len(df)), df['Country'])
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df['TotalSales'])):
            plt.text(bar.get_width() + max(df['TotalSales'])*0.01, bar.get_y() + bar.get_height()/2,
                    f'${value:,.0f}', ha='left', va='center', fontweight='bold')
        
        # Add grid
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        output_path = '../images/geographic_sales_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Geographic analysis chart saved to: {output_path}")
        
        plt.show()
        return True
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report"""
        logger.info("Generating analysis report")
        
        # Execute key queries for analysis
        queries = {
            'Total Sales': "SELECT SUM(TotalSales) as TotalSales FROM SalesFact",
            'Total Transactions': "SELECT COUNT(*) as TransactionCount FROM SalesFact",
            'Unique Customers': "SELECT COUNT(DISTINCT CustomerID) as CustomerCount FROM SalesFact",
            'Sales by Country': """
                SELECT l.Country, SUM(f.TotalSales) as TotalSales
                FROM SalesFact f
                JOIN LocationDim l ON f.LocationID = l.LocationID
                GROUP BY l.Country
                ORDER BY TotalSales DESC
                LIMIT 5
            """,
            'Top Product Categories': """
                SELECT p.Category, SUM(f.TotalSales) as TotalSales
                FROM SalesFact f
                JOIN ProductDim p ON f.ProductID = p.ProductID
                GROUP BY p.Category
                ORDER BY TotalSales DESC
                LIMIT 5
            """
        }
        
        report_data = {}
        for name, query in queries.items():
        df = self.execute_query(query)
            if df is not None and not df.empty:
                report_data[name] = df
        
        # Generate report
        report = f"""
# OLAP Analysis Report
## DSA 2040 Practical Exam - Task 3

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Executive Summary
This report provides comprehensive analysis of the retail data warehouse using OLAP operations.

### Key Metrics
- **Total Sales:** ${report_data.get('Total Sales', pd.DataFrame()).iloc[0, 0] if not report_data.get('Total Sales', pd.DataFrame()).empty else 'N/A':,.2f}
- **Total Transactions:** {report_data.get('Total Transactions', pd.DataFrame()).iloc[0, 0] if not report_data.get('Total Transactions', pd.DataFrame()).empty else 'N/A':,}
- **Unique Customers:** {report_data.get('Unique Customers', pd.DataFrame()).iloc[0, 0] if not report_data.get('Unique Customers', pd.DataFrame()).empty else 'N/A':,}

### Top Performing Countries
"""
        
        if 'Sales by Country' in report_data:
            for _, row in report_data['Sales by Country'].iterrows():
                report += f"- **{row['Country']}:** ${row['TotalSales']:,.2f}\n"
        
        report += """
### Top Product Categories
"""
        
        if 'Top Product Categories' in report_data:
            for _, row in report_data['Top Product Categories'].iterrows():
                report += f"- **{row['Category']}:** ${row['TotalSales']:,.2f}\n"
        
        report += """
### Business Insights

1. **Geographic Performance:** The data shows varying performance across different countries, indicating regional market differences.

2. **Product Mix:** Different product categories show varying levels of success, providing insights for inventory management.

3. **Customer Base:** The number of unique customers and transaction patterns reveal customer behavior insights.

4. **Temporal Trends:** Time-based analysis shows seasonal patterns and growth trends.

### Recommendations

1. **Focus on High-Performing Markets:** Concentrate resources on countries showing strong sales performance.

2. **Product Strategy:** Develop strategies for underperforming product categories or expand successful ones.

3. **Customer Engagement:** Analyze customer segments to improve retention and acquisition strategies.

4. **Operational Efficiency:** Use transaction patterns to optimize inventory and staffing.

### Technical Notes

- This analysis is based on synthetic data generated for educational purposes
- The star schema design enables efficient querying across multiple dimensions
- All visualizations are saved in the images/ directory
- The data warehouse supports real-time analytical queries for business intelligence

---
*Report generated automatically by OLAP Visualizer*
"""
        
        # Save report
        output_path = '../reports/olap_analysis_report.md'
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Analysis report saved to: {output_path}")
        return report
        
    def run_all_visualizations(self):
        """Run all visualization functions"""
        logger.info("Starting all OLAP visualizations")
        
        try:
            # Connect to database
            if not self.connect_database():
                logger.error("Failed to connect to database")
                return False
            
            # Create all visualizations
            results = []
            
            results.append(self.create_sales_by_country_chart())
            results.append(self.create_time_series_analysis())
            results.append(self.create_customer_segmentation_chart())
            results.append(self.create_product_performance_chart())
            results.append(self.create_geographic_analysis_chart())
            
            # Generate analysis report
            self.generate_analysis_report()
            
            # Summary
            successful = sum(results)
            total = len(results)
            logger.info(f"Visualization summary: {successful}/{total} charts created successfully")
            
            return successful == total
            
        except Exception as e:
            logger.error(f"Error during visualization process: {str(e)}")
            return False
        
        finally:
        if self.conn:
            self.conn.close()
                logger.info("Database connection closed")

def main():
    """Main function to run all visualizations"""
    try:
        # Initialize visualizer
        visualizer = OLAPVisualizer()
        
        # Run all visualizations
        success = visualizer.run_all_visualizations()
        
        if success:
            logger.info("All OLAP visualizations completed successfully!")
            print("✅ All visualizations created successfully!")
            print("📊 Check the images/ directory for charts")
            print("📋 Check the reports/ directory for analysis report")
        else:
            logger.warning("Some visualizations failed")
            print("⚠️ Some visualizations failed. Check logs for details.")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
