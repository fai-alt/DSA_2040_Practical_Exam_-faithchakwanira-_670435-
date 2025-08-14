-- DSA 2040 Practical Exam - Task 3: OLAP Queries and Analysis
-- OLAP Queries for Retail Data Warehouse
-- Author: [Your Name]
-- Date: August 13, 2025

-- ==========================================
-- OLAP QUERIES FOR RETAIL ANALYSIS
-- ==========================================

-- Query 1: ROLL-UP - Total Sales by Country and Quarter
-- This query demonstrates roll-up operation by aggregating sales from transaction level to country and quarter level
SELECT 
    l.Country,
    t.Year,
    t.Quarter,
    SUM(f.TotalSales) as TotalSales,
    COUNT(f.SaleID) as TransactionCount,
    AVG(f.TotalSales) as AverageSaleValue
FROM SalesFact f
JOIN LocationDim l ON f.LocationID = l.LocationID
JOIN TimeDim t ON f.TimeID = t.TimeID
GROUP BY l.Country, t.Year, t.Quarter
ORDER BY l.Country, t.Year, t.Quarter;

-- Query 2: DRILL-DOWN - Sales Details for Specific Country by Month
-- This query demonstrates drill-down operation by showing detailed monthly sales for Germany
SELECT 
    l.Country,
    t.Year,
    t.Month,
    t.MonthName,
    SUM(f.TotalSales) as TotalSales,
    COUNT(f.SaleID) as TransactionCount,
    SUM(f.Quantity) as TotalQuantity,
    AVG(f.UnitPrice) as AverageUnitPrice
FROM SalesFact f
JOIN LocationDim l ON f.LocationID = l.LocationID
JOIN TimeDim t ON f.TimeID = t.TimeID
WHERE l.Country = 'Germany'
GROUP BY l.Country, t.Year, t.Month, t.MonthName
ORDER BY t.Year, t.Month;

-- Query 3: SLICE - Total Sales for Electronics Category
-- This query demonstrates slice operation by filtering for specific product category
SELECT 
    p.Category,
    p.Subcategory,
    l.Country,
    t.Year,
    t.Quarter,
    SUM(f.TotalSales) as TotalSales,
    COUNT(f.SaleID) as TransactionCount,
    SUM(f.Quantity) as TotalQuantity
FROM SalesFact f
JOIN ProductDim p ON f.ProductID = p.ProductID
JOIN LocationDim l ON f.LocationID = l.LocationID
JOIN TimeDim t ON f.TimeID = t.TimeID
WHERE p.Category = 'Electronics'
GROUP BY p.Category, p.Subcategory, l.Country, t.Year, t.Quarter
ORDER BY l.Country, t.Year, t.Quarter;

-- Query 4: Additional Analysis - Customer Segmentation by Sales
-- This query analyzes customer behavior and segments
SELECT 
    c.CustomerSegment,
    c.Country,
    COUNT(DISTINCT f.CustomerID) as CustomerCount,
    SUM(f.TotalSales) as TotalSales,
    AVG(f.TotalSales) as AverageCustomerSales,
    COUNT(f.SaleID) as TotalTransactions
FROM SalesFact f
JOIN CustomerDim c ON f.CustomerID = c.CustomerID
GROUP BY c.CustomerSegment, c.Country
ORDER BY c.Country, TotalSales DESC;

-- Query 5: Time-based Trend Analysis
-- This query shows sales trends over time
SELECT 
    t.Year,
    t.Month,
    t.MonthName,
    SUM(f.TotalSales) as TotalSales,
    COUNT(f.SaleID) as TransactionCount,
    SUM(f.Quantity) as TotalQuantity,
    LAG(SUM(f.TotalSales)) OVER (ORDER BY t.Year, t.Month) as PreviousMonthSales,
    ((SUM(f.TotalSales) - LAG(SUM(f.TotalSales)) OVER (ORDER BY t.Year, t.Month)) / 
     LAG(SUM(f.TotalSales)) OVER (ORDER BY t.Year, t.Month) * 100) as MonthOverMonthGrowth
FROM SalesFact f
JOIN TimeDim t ON f.TimeID = t.TimeID
GROUP BY t.Year, t.Month, t.MonthName
ORDER BY t.Year, t.Month;

-- Query 6: Product Performance Analysis
-- This query analyzes product performance across different dimensions
SELECT 
    p.Category,
    p.Subcategory,
    l.Country,
    SUM(f.TotalSales) as TotalSales,
    SUM(f.Quantity) as TotalQuantity,
    AVG(f.UnitPrice) as AverageUnitPrice,
    COUNT(f.SaleID) as TransactionCount,
    RANK() OVER (PARTITION BY p.Category ORDER BY SUM(f.TotalSales) DESC) as CategoryRank
FROM SalesFact f
JOIN ProductDim p ON f.ProductID = p.ProductID
JOIN LocationDim l ON f.LocationID = l.LocationID
GROUP BY p.Category, p.Subcategory, l.Country
ORDER BY p.Category, TotalSales DESC;

-- Query 7: Geographic Sales Distribution
-- This query shows sales distribution across different geographic regions
SELECT 
    l.Region,
    l.Country,
    l.City,
    SUM(f.TotalSales) as TotalSales,
    COUNT(f.SaleID) as TransactionCount,
    COUNT(DISTINCT f.CustomerID) as UniqueCustomers,
    AVG(f.TotalSales) as AverageTransactionValue
FROM SalesFact f
JOIN LocationDim l ON f.LocationID = l.LocationID
GROUP BY l.Region, l.Country, l.City
ORDER BY l.Region, TotalSales DESC;

-- Query 8: Weekend vs Weekday Sales Analysis
-- This query analyzes sales patterns on weekends vs weekdays
SELECT 
    t.IsWeekend,
    CASE 
        WHEN t.IsWeekend = 1 THEN 'Weekend'
        ELSE 'Weekday'
    END as DayType,
    COUNT(f.SaleID) as TransactionCount,
    SUM(f.TotalSales) as TotalSales,
    AVG(f.TotalSales) as AverageTransactionValue,
    SUM(f.Quantity) as TotalQuantity
FROM SalesFact f
JOIN TimeDim t ON f.TimeID = t.TimeID
GROUP BY t.IsWeekend
ORDER BY t.IsWeekend;

-- ==========================================
-- QUERY EXPLANATIONS
-- ==========================================

/*
ROLL-UP (Query 1): Aggregates sales data from individual transactions to country and quarter level,
showing higher-level business insights.

DRILL-DOWN (Query 2): Provides detailed monthly breakdown for a specific country (Germany),
allowing analysts to explore specific areas of interest.

SLICE (Query 3): Filters data for a specific product category (Electronics) across all other dimensions,
enabling focused analysis on particular business segments.

These queries demonstrate the power of the star schema design in supporting various analytical needs
and business intelligence requirements.
*/
