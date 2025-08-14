-- DSA 2040 Practical Exam - Task 1: Data Warehouse Design
-- SQL Implementation for Star Schema
-- Author: [Your Name]
-- Date: August 13, 2025

-- Drop existing tables if they exist
DROP TABLE IF EXISTS SalesFact;
DROP TABLE IF EXISTS CustomerDim;
DROP TABLE IF EXISTS ProductDim;
DROP TABLE IF EXISTS TimeDim;
DROP TABLE IF EXISTS LocationDim;

-- Create Dimension Tables

-- Customer Dimension
CREATE TABLE CustomerDim (
    CustomerID INTEGER PRIMARY KEY,
    CustomerName VARCHAR(100),
    CustomerSegment VARCHAR(50),
    Country VARCHAR(100),
    Region VARCHAR(50),
    CustomerType VARCHAR(20)
);

-- Product Dimension
CREATE TABLE ProductDim (
    ProductID INTEGER PRIMARY KEY,
    StockCode VARCHAR(20),
    Description VARCHAR(200),
    Category VARCHAR(50),
    Subcategory VARCHAR(50),
    Brand VARCHAR(50)
);

-- Time Dimension
CREATE TABLE TimeDim (
    TimeID INTEGER PRIMARY KEY,
    InvoiceDate DATE,
    Year INTEGER,
    Quarter INTEGER,
    Month INTEGER,
    MonthName VARCHAR(20),
    DayOfWeek INTEGER,
    DayName VARCHAR(20),
    IsWeekend BOOLEAN
);

-- Location Dimension
CREATE TABLE LocationDim (
    LocationID INTEGER PRIMARY KEY,
    Country VARCHAR(100),
    Region VARCHAR(50),
    City VARCHAR(100)
);

-- Create Fact Table
CREATE TABLE SalesFact (
    SaleID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    ProductID INTEGER,
    TimeID INTEGER,
    LocationID INTEGER,
    InvoiceNo VARCHAR(20),
    Quantity INTEGER,
    UnitPrice DECIMAL(10,2),
    TotalSales DECIMAL(10,2),
    FOREIGN KEY (CustomerID) REFERENCES CustomerDim(CustomerID),
    FOREIGN KEY (ProductID) REFERENCES ProductDim(ProductID),
    FOREIGN KEY (TimeID) REFERENCES TimeDim(TimeID),
    FOREIGN KEY (LocationID) REFERENCES LocationDim(LocationID)
);

-- Create indexes for better query performance
CREATE INDEX idx_sales_customer ON SalesFact(CustomerID);
CREATE INDEX idx_sales_product ON SalesFact(ProductID);
CREATE INDEX idx_sales_time ON SalesFact(TimeID);
CREATE INDEX idx_sales_location ON SalesFact(LocationID);
CREATE INDEX idx_sales_date ON SalesFact(TimeID);
CREATE INDEX idx_customer_country ON CustomerDim(Country);
CREATE INDEX idx_product_category ON ProductDim(Category);
CREATE INDEX idx_time_year ON TimeDim(Year);
CREATE INDEX idx_time_quarter ON TimeDim(Quarter);
CREATE INDEX idx_time_month ON TimeDim(Month);

-- Insert sample data for demonstration
INSERT INTO CustomerDim (CustomerID, CustomerName, CustomerSegment, Country, Region, CustomerType) VALUES
(1001, 'John Smith', 'Premium', 'United Kingdom', 'Europe', 'Individual'),
(1002, 'Maria Garcia', 'Standard', 'Spain', 'Europe', 'Individual'),
(1003, 'TechCorp Ltd', 'Enterprise', 'Germany', 'Europe', 'Business');

INSERT INTO ProductDim (ProductID, StockCode, Description, Category, Subcategory, Brand) VALUES
(1, 'PROD1001', 'Smartphone Product 1', 'Electronics', 'Smartphones', 'TechBrand'),
(2, 'PROD1002', 'Laptop Product 2', 'Electronics', 'Laptops', 'TechBrand'),
(3, 'PROD1003', 'Men Clothing Product 3', 'Clothing', 'Men', 'FashionBrand');

INSERT INTO LocationDim (LocationID, Country, Region, City) VALUES
(1, 'United Kingdom', 'Europe', 'London'),
(2, 'Germany', 'Europe', 'Berlin'),
(3, 'Spain', 'Europe', 'Madrid');

-- Note: TimeDim and SalesFact will be populated by the ETL process
