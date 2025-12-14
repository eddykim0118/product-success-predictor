"""
Data Exploration Script
=======================

Explore Amazon Beauty dataset to understand structure and define prediction task.
"""

import sys
sys.path.append('..')

from src.data.loader import AmazonBeautyDataLoader
import pandas as pd
import numpy as np


def main():
    """Explore the dataset."""
    
    print("="*70)
    print("AMAZON BEAUTY DATASET EXPLORATION")
    print("="*70)
    
    # Load data (start with 10K rows)
    loader = AmazonBeautyDataLoader()
    df = loader.load_data(nrows=10000)
    
    # Basic info
    print("\n" + "="*70)
    print("BASIC INFO")
    print("="*70)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    
    # Missing values
    print("\n" + "="*70)
    print("MISSING VALUES")
    print("="*70)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0])
    
    # First few rows
    print("\n" + "="*70)
    print("SAMPLE ROWS")
    print("="*70)
    print(df.head(10))
    
    # Numeric columns stats
    print("\n" + "="*70)
    print("NUMERIC COLUMNS STATISTICS")
    print("="*70)
    print(df.describe())
    
    # Unique values for categorical columns
    print("\n" + "="*70)
    print("CATEGORICAL COLUMNS")
    print("="*70)
    for col in df.select_dtypes(include=['object']).columns:
        n_unique = df[col].nunique()
        print(f"{col}: {n_unique:,} unique values")
        if n_unique < 10:
            print(f"  Values: {df[col].unique()}")
    
    print("\n" + "="*70)
    print("EXPLORATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()