"""
Test data preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import CrimeDataPreprocessor
from src.utils import create_sample_crime_data, create_sample_demographic_data

class TestDataPreprocessing:
    """Test cases for data preprocessing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = CrimeDataPreprocessor()
        self.sample_crime_data = create_sample_crime_data(100)
        self.sample_demographic_data = create_sample_demographic_data()
    
    def test_load_data(self):
        """Test data loading functionality."""
        # Test with sample dataframe
        df = self.sample_crime_data.copy()
        assert not df.empty
        assert len(df) == 100
        assert 'crime_type' in df.columns
        assert 'state' in df.columns
    
    def test_clean_missing_values(self):
        """Test missing value cleaning."""
        # Create data with missing values
        df = self.sample_crime_data.copy()
        df.loc[0:5, 'crime_type'] = np.nan
        df.loc[10:15, 'victim_age'] = np.nan
        
        # Clean missing values
        cleaned_df = self.preprocessor.clean_missing_values(df)
        
        # Check that missing values are handled
        assert cleaned_df.isnull().sum().sum() == 0
        assert len(cleaned_df) == len(df)
    
    def test_extract_temporal_features(self):
        """Test temporal feature extraction."""
        df = self.sample_crime_data.copy()
        
        # Extract temporal features
        temporal_df = self.preprocessor.extract_temporal_features(df)
        
        # Check that temporal features are created
        expected_features = ['year', 'month', 'day', 'weekday', 'weekend', 
                           'quarter', 'day_of_year', 'week_of_year', 'is_holiday']
        
        for feature in expected_features:
            assert feature in temporal_df.columns
        
        # Check data types and ranges
        assert temporal_df['weekend'].dtype == int
        assert temporal_df['weekend'].isin([0, 1]).all()
        assert temporal_df['month'].between(1, 12).all()
        assert temporal_df['weekday'].between(0, 6).all()
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        df = self.sample_crime_data.copy()
        categorical_columns = ['crime_type', 'state', 'case_status']
        
        # Encode categorical features
        encoded_df = self.preprocessor.encode_categorical_features(df, categorical_columns)
        
        # Check that encoded columns are created
        for col in categorical_columns:
            if col in df.columns:
                # Check for either one-hot encoded or label encoded columns
                encoded_found = any(
                    encoded_col.startswith(f"{col}_") or encoded_col == f"{col}_encoded" 
                    for encoded_col in encoded_df.columns
                )
                assert encoded_found, f"No encoded version found for {col}"
    
    def test_merge_demographic_data(self):
        """Test demographic data merging."""
        crime_df = self.sample_crime_data.copy()
        demo_df = self.sample_demographic_data.copy()
        
        # Merge datasets
        merged_df = self.preprocessor.merge_demographic_data(crime_df, demo_df)
        
        # Check merge results
        assert len(merged_df) == len(crime_df)
        assert 'population' in merged_df.columns
        assert 'literacy_rate' in merged_df.columns
        assert not merged_df.empty
    
    def test_create_aggregated_features(self):
        """Test aggregated feature creation."""
        df = self.sample_crime_data.copy()
        
        # Create aggregated features
        agg_df = self.preprocessor.create_aggregated_features(df)
        
        # Check that aggregated features are created
        expected_agg_features = ['state_crime_count', 'district_crime_count', 'crime_type_count']
        
        for feature in expected_agg_features:
            if feature.replace('_count', '') in df.columns:
                assert feature in agg_df.columns
                assert agg_df[feature].dtype in [int, float]
                assert (agg_df[feature] > 0).all()
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        crime_df = self.sample_crime_data.copy()
        demo_df = self.sample_demographic_data.copy()
        
        # Run complete pipeline
        processed_df = self.preprocessor.preprocess_pipeline(crime_df, demo_df)
        
        # Check pipeline results
        assert not processed_df.empty
        assert processed_df.shape[1] > crime_df.shape[1]  # More features added
        assert processed_df.isnull().sum().sum() == 0  # No missing values
        
        # Check that key features exist
        assert 'year' in processed_df.columns
        assert 'month' in processed_df.columns
        assert 'population' in processed_df.columns
    
    def test_get_preprocessing_summary(self):
        """Test preprocessing summary generation."""
        original_df = self.sample_crime_data.copy()
        processed_df = self.preprocessor.preprocess_pipeline(
            original_df, self.sample_demographic_data
        )
        
        # Get summary
        summary = self.preprocessor.get_preprocessing_summary(original_df, processed_df)
        
        # Check summary contents
        assert 'original_shape' in summary
        assert 'processed_shape' in summary
        assert 'features_added' in summary
        assert 'missing_values_original' in summary
        assert 'missing_values_processed' in summary
        
        # Check summary values
        assert summary['original_shape'] == original_df.shape
        assert summary['processed_shape'] == processed_df.shape
        assert summary['features_added'] >= 0
        assert summary['missing_values_processed'] == 0