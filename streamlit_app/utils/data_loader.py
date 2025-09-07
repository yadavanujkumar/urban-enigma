"""
Data loading utilities for Streamlit app.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import create_sample_crime_data, create_sample_demographic_data
from src.data_preprocessing import CrimeDataPreprocessor

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loading utility for Streamlit dashboard.
    """
    
    def __init__(self):
        """Initialize data loader."""
        self.processed_data_path = "data/processed/crime_data_processed.csv"
        self.sample_data_path = "data/sample_data/crime_data_sample.csv"
        
    def load_processed_data(self) -> pd.DataFrame:
        """
        Load processed crime data for dashboard.
        
        Returns:
            pd.DataFrame: Processed crime data
        """
        try:
            # Try to load processed data first
            if os.path.exists(self.processed_data_path):
                df = pd.read_csv(self.processed_data_path)
                logger.info(f"Loaded processed data: {df.shape}")
                return df
            
            # If processed data doesn't exist, try sample data
            elif os.path.exists(self.sample_data_path):
                df = pd.read_csv(self.sample_data_path)
                logger.info(f"Loaded sample data: {df.shape}")
                return df
            
            # If no data exists, create sample data
            else:
                logger.info("No existing data found, creating sample data...")
                df = self._create_sample_data()
                return df
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Return empty dataframe as fallback
            return pd.DataFrame()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample data for demonstration.
        
        Returns:
            pd.DataFrame: Sample crime data
        """
        try:
            # Create sample datasets
            crime_data = create_sample_crime_data(1000)
            demographic_data = create_sample_demographic_data()
            
            # Process the data
            preprocessor = CrimeDataPreprocessor()
            processed_data = preprocessor.preprocess_pipeline(crime_data, demographic_data)
            
            # Save for future use
            os.makedirs("data/sample_data", exist_ok=True)
            crime_data.to_csv(self.sample_data_path, index=False)
            
            os.makedirs("data/processed", exist_ok=True)
            processed_data.to_csv(self.processed_data_path, index=False)
            
            logger.info(f"Created and saved sample data: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            return pd.DataFrame()
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Summary statistics
        """
        if df.empty:
            return {}
        
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'date_range': self._get_date_range(df),
            'states_covered': df['state'].nunique() if 'state' in df.columns else 0,
            'districts_covered': df['district'].nunique() if 'district' in df.columns else 0,
            'crime_types': df['crime_type'].nunique() if 'crime_type' in df.columns else 0,
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return summary
    
    def _get_date_range(self, df: pd.DataFrame) -> str:
        """
        Get date range from dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            str: Date range string
        """
        try:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                min_date = df['date'].min().strftime('%Y-%m-%d')
                max_date = df['date'].max().strftime('%Y-%m-%d')
                return f"{min_date} to {max_date}"
            elif 'year' in df.columns:
                min_year = df['year'].min()
                max_year = df['year'].max()
                return f"{min_year} to {max_year}"
            else:
                return "Unknown"
        except Exception:
            return "Unknown"
    
    def filter_data(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """
        Apply filters to dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            filters (dict): Filter criteria
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        filtered_df = df.copy()
        
        try:
            # State filter
            if 'states' in filters and filters['states']:
                if 'state' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['state'].isin(filters['states'])]
            
            # Year range filter
            if 'year_range' in filters and filters['year_range']:
                if 'year' in filtered_df.columns:
                    min_year, max_year = filters['year_range']
                    filtered_df = filtered_df[
                        (filtered_df['year'] >= min_year) & 
                        (filtered_df['year'] <= max_year)
                    ]
            
            # Crime type filter
            if 'crime_types' in filters and filters['crime_types']:
                if 'crime_type' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['crime_type'].isin(filters['crime_types'])]
            
            # District filter
            if 'districts' in filters and filters['districts']:
                if 'district' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['district'].isin(filters['districts'])]
            
            logger.info(f"Applied filters, resulting shape: {filtered_df.shape}")
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return df
        
        return filtered_df
    
    def get_geographic_bounds(self, df: pd.DataFrame) -> dict:
        """
        Get geographic bounds for mapping.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Geographic bounds
        """
        bounds = {}
        
        try:
            if 'latitude' in df.columns and 'longitude' in df.columns:
                bounds = {
                    'min_lat': df['latitude'].min(),
                    'max_lat': df['latitude'].max(),
                    'min_lon': df['longitude'].min(),
                    'max_lon': df['longitude'].max(),
                    'center_lat': df['latitude'].mean(),
                    'center_lon': df['longitude'].mean()
                }
        except Exception as e:
            logger.error(f"Error calculating geographic bounds: {e}")
        
        return bounds
    
    def prepare_time_series_data(self, df: pd.DataFrame, freq: str = 'M') -> pd.Series:
        """
        Prepare time series data for forecasting.
        
        Args:
            df (pd.DataFrame): Input dataframe
            freq (str): Frequency for aggregation
            
        Returns:
            pd.Series: Time series data
        """
        try:
            if 'date' not in df.columns:
                logger.warning("No date column found for time series")
                return pd.Series()
            
            df_ts = df.copy()
            df_ts['date'] = pd.to_datetime(df_ts['date'])
            df_ts = df_ts.set_index('date')
            
            # Aggregate by frequency
            if freq == 'D':
                ts = df_ts.resample('D').size()
            elif freq == 'W':
                ts = df_ts.resample('W').size()
            elif freq == 'M':
                ts = df_ts.resample('M').size()
            elif freq == 'Q':
                ts = df_ts.resample('Q').size()
            elif freq == 'Y':
                ts = df_ts.resample('Y').size()
            else:
                ts = df_ts.resample('M').size()
            
            ts = ts.fillna(0)
            
            logger.info(f"Prepared time series data: {len(ts)} periods")
            return ts
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            return pd.Series()
    
    def get_top_items(self, df: pd.DataFrame, column: str, n: int = 10) -> pd.Series:
        """
        Get top N items from a column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name
            n (int): Number of top items
            
        Returns:
            pd.Series: Top items with counts
        """
        try:
            if column in df.columns:
                return df[column].value_counts().head(n)
            else:
                return pd.Series()
        except Exception as e:
            logger.error(f"Error getting top items for {column}: {e}")
            return pd.Series()