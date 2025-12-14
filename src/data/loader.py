"""
Data Loader
===========

Load and prepare Amazon Beauty dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class AmazonBeautyDataLoader:
    """
    Load Amazon Beauty Product data.
    """
    
    def __init__(self, data_path='data/raw/Amazon_Beauty_Recommendation.csv'):
        """
        Args:
            data_path: Path to CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
    
    def load_data(self, nrows=None):
        """
        Load CSV data.
        
        Args:
            nrows: Number of rows to load (None = all)
            
        Returns:
            DataFrame
        """
        print(f"Loading data from {self.data_path}...")
        
        self.df = pd.read_csv(self.data_path, nrows=nrows)
        
        print(f"Loaded {len(self.df):,} rows")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def get_info(self):
        """
        Print dataset info.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*70)
        print("DATASET INFO")
        print("="*70)
        print(f"Shape: {self.df.shape}")
        print(f"\nColumn types:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\nFirst few rows:")
        print(self.df.head())
        print("="*70)
    
    def get_sample(self, n=1000):
        """
        Get random sample.
        
        Args:
            n: Sample size
            
        Returns:
            DataFrame sample
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df.sample(n=min(n, len(self.df)), random_state=42)