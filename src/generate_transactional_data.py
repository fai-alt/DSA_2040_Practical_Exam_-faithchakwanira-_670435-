#!/usr/bin/env python3
"""
Synthetic Transactional Data Generator for DSA 2040 Practical Exam
Generates 20-50 transactions for association rule mining
"""

import pandas as pd
import numpy as np
import random
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_transactions(n_transactions=40, min_items=3, max_items=8):
    """
    Generate synthetic transactional data with realistic patterns
    
    Args:
        n_transactions (int): Number of transactions to generate (default: 40)
        min_items (int): Minimum items per transaction (default: 3)
        max_items (int): Maximum items per transaction (default: 8)
    
    Returns:
        list: List of transactions (each transaction is a list of items)
    """
    
    # Pool of 20 items (common retail products)
    items_pool = [
        'milk', 'bread', 'beer', 'diapers', 'eggs', 'cheese', 'yogurt', 'bananas',
        'apples', 'chicken', 'beef', 'rice', 'pasta', 'tomatoes', 'onions',
        'potatoes', 'carrots', 'coffee', 'tea', 'sugar'
    ]
    
    # Define some common co-occurrence patterns (realistic retail behavior)
    common_patterns = [
        ['milk', 'bread'],           # Breakfast items
        ['milk', 'cereal'],          # Breakfast items
        ['beer', 'chips'],           # Party items
        ['diapers', 'wipes'],        # Baby care
        ['eggs', 'milk'],            # Baking/breakfast
        ['chicken', 'rice'],         # Meal preparation
        ['beef', 'potatoes'],        # Meal preparation
        ['tomatoes', 'onions'],      # Cooking ingredients
        ['coffee', 'sugar'],         # Coffee preparation
        ['bread', 'cheese'],         # Sandwich making
        ['pasta', 'tomatoes'],       # Italian cooking
        ['bananas', 'yogurt'],       # Healthy snack
        ['apples', 'peanut_butter'], # Healthy snack
        ['beer', 'nuts'],            # Party snacks
        ['milk', 'cookies'],         # Kids' snack
    ]
    
    # Add some items that don't exist in our pool but are in patterns
    # We'll replace them with items from our pool
    pattern_replacements = {
        'cereal': 'rice',
        'wipes': 'sugar',
        'chips': 'potatoes',
        'nuts': 'carrots',
        'cookies': 'apples',
        'peanut_butter': 'cheese'
    }
    
    # Clean up patterns to only use items from our pool
    cleaned_patterns = []
    for pattern in common_patterns:
        cleaned_pattern = []
        for item in pattern:
            if item in items_pool:
                cleaned_pattern.append(item)
            elif item in pattern_replacements:
                cleaned_pattern.append(pattern_replacements[item])
        if len(cleaned_pattern) >= 2:
            cleaned_patterns.append(cleaned_pattern)
    
    transactions = []
    
    for i in range(n_transactions):
        # Determine transaction size
        transaction_size = random.randint(min_items, max_items)
        
        # Start with a random common pattern (70% chance)
        if random.random() < 0.7 and cleaned_patterns:
            pattern = random.choice(cleaned_patterns)
            transaction = pattern.copy()
            remaining_items = transaction_size - len(pattern)
        else:
            transaction = []
            remaining_items = transaction_size
        
        # Fill remaining items randomly
        available_items = [item for item in items_pool if item not in transaction]
        
        # Add some items with higher probability for realistic patterns
        for _ in range(remaining_items):
            if available_items:
                # 60% chance to pick from high-frequency items
                if random.random() < 0.6 and len(available_items) > 5:
                    # Pick from first 10 items (more common items)
                    item = random.choice(available_items[:10])
                else:
                    item = random.choice(available_items)
                
                transaction.append(item)
                available_items.remove(item)
        
        # Shuffle the transaction for realism
        random.shuffle(transaction)
        transactions.append(transaction)
    
    return transactions

def create_transaction_dataframe(transactions):
    """
    Convert transactions to a format suitable for analysis
    
    Args:
        transactions (list): List of transactions
    
    Returns:
        pd.DataFrame: DataFrame with transaction data
    """
    
    # Create a list of dictionaries for DataFrame
    data = []
    for i, transaction in enumerate(transactions):
        for item in transaction:
            data.append({
                'TransactionID': i + 1,
                'Item': item
            })
    
    return pd.DataFrame(data)

def analyze_transaction_patterns(transactions):
    """
    Analyze the generated transaction patterns
    
    Args:
        transactions (list): List of transactions
    """
    
    print(f"\nTransaction Analysis:")
    print(f"Total transactions: {len(transactions)}")
    print(f"Total items: {sum(len(t) for t in transactions)}")
    print(f"Average items per transaction: {sum(len(t) for t in transactions) / len(transactions):.2f}")
    
    # Count item frequencies
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    
    print(f"\nTop 10 most frequent items:")
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    for item, count in sorted_items[:10]:
        support = count / len(transactions)
        print(f"  {item}: {count} times (support: {support:.3f})")
    
    # Find some common co-occurrences
    print(f"\nSample co-occurrence patterns:")
    for i, transaction in enumerate(transactions[:5]):
        print(f"  Transaction {i+1}: {', '.join(transaction)}")

def save_transactions_to_file(transactions, output_file):
    """
    Save transactions to a text file for easy reading
    
    Args:
        transactions (list): List of transactions
        output_file (str): Output file path
    """
    
    with open(output_file, 'w') as f:
        f.write("# Synthetic Transactional Data for DSA 2040 Practical Exam\n")
        f.write(f"# Generated {len(transactions)} transactions\n")
        f.write("# Each line represents one transaction with items separated by commas\n\n")
        
        for i, transaction in enumerate(transactions):
            f.write(f"Transaction_{i+1}: {','.join(transaction)}\n")
    
    print(f"Transactions saved to: {output_file}")

def main():
    """Main function to generate and save synthetic transactional data"""
    
    print("Generating synthetic transactional data for DSA 2040 Practical Exam...")
    
    # Generate transactions
    transactions = generate_synthetic_transactions(40, 3, 8)
    
    # Display sample transactions
    print(f"\nGenerated {len(transactions)} transactions")
    print("\nSample transactions:")
    for i, transaction in enumerate(transactions[:10]):
        print(f"  {i+1:2d}: {', '.join(transaction)}")
    
    # Analyze patterns
    analyze_transaction_patterns(transactions)
    
    # Create DataFrame
    df = create_transaction_dataframe(transactions)
    
    # Save to CSV
    csv_output = '../data/synthetic_transactions.csv'
    df.to_csv(csv_output, index=False)
    print(f"\nTransaction data saved to: {csv_output}")
    
    # Save to readable text file
    txt_output = '../data/synthetic_transactions.txt'
    save_transactions_to_file(transactions, txt_output)
    
    # Display DataFrame info
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Unique items: {df['Item'].nunique()}")
    print(f"Unique transactions: {df['TransactionID'].nunique()}")
    
    return transactions, df

if __name__ == "__main__":
    transactions, df = main()
