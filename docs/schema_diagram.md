# Data Warehouse Schema Design - Star Schema

## Overview
This document describes the star schema design implemented for the retail data warehouse as part of the DSA 2040 Practical Exam.

## Schema Diagram

```
                    ┌─────────────────┐
                    │   SalesFact     │
                    │                 │
                    │ SaleID (PK)     │
                    │ CustomerID (FK) │
                    │ ProductID (FK)  │
                    │ TimeID (FK)     │
                    │ LocationID (FK) │
                    │ InvoiceNo       │
                    │ Quantity        │
                    │ UnitPrice       │
                    │ TotalSales      │
                    └─────────────────┘
                            │
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        │                   │                   │
┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
│ CustomerDim  │    │  ProductDim  │    │   TimeDim    │
│              │    │              │    │              │
│ CustomerID   │    │ ProductID    │    │ TimeID       │
│ CustomerName │    │ StockCode    │    │ InvoiceDate  │
│ Segment      │    │ Description  │    │ Year         │
│ Country      │    │ Category     │    │ Quarter      │
│ Region       │    │ Subcategory  │    │ Month        │
│ CustomerType │    │ Brand        │    │ MonthName    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼──────┐
                    │ LocationDim  │
                    │              │
                    │ LocationID   │
                    │ Country      │
                    │ Region       │
                    │ City         │
                    └──────────────┘
```

## Star Schema vs Snowflake Schema

**Why Star Schema over Snowflake Schema:**

1. **Simplicity**: Star schema is simpler to understand and implement, making it ideal for this exam scenario where we need to demonstrate clear data warehouse concepts.

2. **Query Performance**: Star schema typically provides better query performance for OLAP operations since it requires fewer joins between dimension tables.

3. **Maintenance**: Easier to maintain and modify compared to snowflake schema which has more normalized dimension tables.

4. **Business User Friendly**: The flat structure of dimension tables makes it easier for business users to understand and query the data warehouse.

5. **Exam Requirements**: The exam specifically asks for a star schema with 1 fact table and 3-4 dimension tables, which aligns perfectly with our implementation.

## Schema Components

### Fact Table: SalesFact
- **Purpose**: Central table containing business metrics (sales transactions)
- **Measures**: Quantity, UnitPrice, TotalSales
- **Foreign Keys**: Links to all dimension tables
- **Granularity**: One row per sales transaction

### Dimension Tables

#### CustomerDim
- Customer demographics and segmentation
- Supports customer analysis and segmentation

#### ProductDim
- Product information and categorization
- Enables product performance analysis

#### TimeDim
- Time-based analysis (year, quarter, month, day)
- Supports temporal trend analysis

#### LocationDim
- Geographic information for regional analysis
- Enables location-based insights

## Benefits of This Design

1. **Fast Queries**: Optimized for analytical queries with proper indexing
2. **Scalability**: Can handle large volumes of sales data
3. **Flexibility**: Supports various business intelligence requirements
4. **Standards Compliance**: Follows data warehouse best practices
5. **Exam Alignment**: Meets all requirements specified in the practical exam
