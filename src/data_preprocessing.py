"""
Data preprocessing module for crime hotspot prediction.
Handles data cleaning, feature engineering, and transformation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.utils import load_config, ensure_dir_exists, validate_dataframe, get_indian_holidays

logger = logging.getLogger(__name__)

class CrimeDataPreprocessor:
    """
    Comprehensive data preprocessor for crime datasets.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = load_config(config_path)
        self.processed_data = None
        self.feature_columns = []
        self.categorical_encoders = {}
        
    def load_data(self, file_path: str, file_type: str = "csv") -> pd.DataFrame:
        """
        Load crime data from file.
        
        Args:
            file_path (str): Path to data file
            file_type (str): Type of file ('csv', 'excel', 'json')
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            if file_type.lower() == "csv":
                df = pd.read_csv(file_path)
            elif file_type.lower() in ["excel", "xlsx"]:
                df = pd.read_excel(file_path)
            elif file_type.lower() == "json":
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"Loaded data with shape {df.shape} from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_cleaned = df.copy()
        
        logger.info(f"Missing values before cleaning:\n{df.isnull().sum()}")
        
        # Handle missing values based on column type
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().sum() > 0:
                if df_cleaned[column].dtype in ['object']:
                    # Fill categorical missing values with mode
                    mode_value = df_cleaned[column].mode()
                    if len(mode_value) > 0:
                        df_cleaned[column].fillna(mode_value[0], inplace=True)
                    else:
                        df_cleaned[column].fillna('Unknown', inplace=True)
                else:
                    # Fill numerical missing values with median
                    median_value = df_cleaned[column].median()
                    df_cleaned[column].fillna(median_value, inplace=True)
        
        logger.info(f"Missing values after cleaning:\n{df_cleaned.isnull().sum()}")
        return df_cleaned
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], method: str = "iqr") -> pd.DataFrame:
        """
        Remove outliers from specified columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to check for outliers
            method (str): Method to use ('iqr' or 'zscore')
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        df_cleaned = df.copy()
        initial_shape = df_cleaned.shape
        
        for column in columns:
            if column in df_cleaned.columns and df_cleaned[column].dtype in ['int64', 'float64']:
                if method == "iqr":
                    Q1 = df_cleaned[column].quantile(0.25)
                    Q3 = df_cleaned[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & 
                                          (df_cleaned[column] <= upper_bound)]
                elif method == "zscore":
                    z_scores = np.abs((df_cleaned[column] - df_cleaned[column].mean()) / df_cleaned[column].std())
                    df_cleaned = df_cleaned[z_scores < 3]
        
        final_shape = df_cleaned.shape
        removed_rows = initial_shape[0] - final_shape[0]
        logger.info(f"Removed {removed_rows} outliers using {method} method")
        
        return df_cleaned
    
    def extract_temporal_features(self, df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
        """
        Extract temporal features from date column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            date_column (str): Name of date column
            
        Returns:
            pd.DataFrame: Dataframe with temporal features
        """
        df_temporal = df.copy()
        
        if date_column not in df_temporal.columns:
            logger.warning(f"Date column '{date_column}' not found")
            return df_temporal
        
        # Ensure date column is datetime
        df_temporal[date_column] = pd.to_datetime(df_temporal[date_column])
        
        # Extract temporal features
        df_temporal['year'] = df_temporal[date_column].dt.year
        df_temporal['month'] = df_temporal[date_column].dt.month
        df_temporal['day'] = df_temporal[date_column].dt.day
        df_temporal['weekday'] = df_temporal[date_column].dt.dayofweek
        df_temporal['weekend'] = df_temporal['weekday'].isin([5, 6]).astype(int)
        df_temporal['quarter'] = df_temporal[date_column].dt.quarter
        df_temporal['day_of_year'] = df_temporal[date_column].dt.dayofyear
        df_temporal['week_of_year'] = df_temporal[date_column].dt.isocalendar().week
        
        # Add holiday information
        df_temporal['is_holiday'] = 0
        for year in df_temporal['year'].unique():
            holidays = get_indian_holidays(year)
            holiday_dates = pd.to_datetime(holidays)
            df_temporal.loc[df_temporal[date_column].dt.date.isin(holiday_dates.date), 'is_holiday'] = 1
        
        # Add time-based categorical features
        df_temporal['month_name'] = df_temporal[date_column].dt.month_name()
        df_temporal['weekday_name'] = df_temporal[date_column].dt.day_name()
        
        # Season mapping
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
        df_temporal['season'] = df_temporal['month'].map(season_map)
        
        logger.info("Extracted temporal features successfully")
        return df_temporal
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using various encoding methods.
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_columns (List[str]): List of categorical columns to encode
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        df_encoded = df.copy()
        
        for column in categorical_columns:
            if column in df_encoded.columns:
                # Use Label Encoding for ordinal features
                if column in ['case_status']:  # Example of ordinal feature
                    le = LabelEncoder()
                    df_encoded[f"{column}_encoded"] = le.fit_transform(df_encoded[column].astype(str))
                    self.categorical_encoders[column] = le
                
                # Use One-Hot Encoding for nominal features with few categories
                elif df_encoded[column].nunique() <= 10:
                    dummies = pd.get_dummies(df_encoded[column], prefix=column)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                # Use Label Encoding for high cardinality features
                else:
                    le = LabelEncoder()
                    df_encoded[f"{column}_encoded"] = le.fit_transform(df_encoded[column].astype(str))
                    self.categorical_encoders[column] = le
        
        logger.info(f"Encoded {len(categorical_columns)} categorical features")
        return df_encoded
    
    def merge_demographic_data(self, crime_df: pd.DataFrame, demographic_df: pd.DataFrame, 
                             merge_key: str = "state") -> pd.DataFrame:
        """
        Merge crime data with demographic information.
        
        Args:
            crime_df (pd.DataFrame): Crime dataset
            demographic_df (pd.DataFrame): Demographic dataset
            merge_key (str): Key to merge on
            
        Returns:
            pd.DataFrame: Merged dataframe
        """
        try:
            merged_df = crime_df.merge(demographic_df, on=merge_key, how='left')
            logger.info(f"Merged datasets on '{merge_key}' column")
            
            # Handle missing values from merge
            merged_df = self.clean_missing_values(merged_df)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            return crime_df
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated features for analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with aggregated features
        """
        df_agg = df.copy()
        
        # Crime frequency by location
        if 'state' in df_agg.columns:
            state_crime_count = df_agg.groupby('state').size().reset_index(name='state_crime_count')
            df_agg = df_agg.merge(state_crime_count, on='state', how='left')
        
        if 'district' in df_agg.columns:
            district_crime_count = df_agg.groupby('district').size().reset_index(name='district_crime_count')
            df_agg = df_agg.merge(district_crime_count, on='district', how='left')
        
        # Crime type frequency
        if 'crime_type' in df_agg.columns:
            crime_type_count = df_agg.groupby('crime_type').size().reset_index(name='crime_type_count')
            df_agg = df_agg.merge(crime_type_count, on='crime_type', how='left')
        
        # Time-based aggregations
        if 'year' in df_agg.columns and 'month' in df_agg.columns:
            monthly_crime_count = df_agg.groupby(['year', 'month']).size().reset_index(name='monthly_crime_count')
            df_agg = df_agg.merge(monthly_crime_count, on=['year', 'month'], how='left')
        
        logger.info("Created aggregated features")
        return df_agg
    
    def normalize_features(self, df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            numerical_columns (List[str]): List of numerical columns to normalize
            
        Returns:
            pd.DataFrame: Dataframe with normalized features
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        df_normalized = df.copy()
        scaler = StandardScaler()
        
        for column in numerical_columns:
            if column in df_normalized.columns:
                df_normalized[f"{column}_normalized"] = scaler.fit_transform(
                    df_normalized[[column]]
                ).flatten()
        
        logger.info(f"Normalized {len(numerical_columns)} numerical features")
        return df_normalized
    
    def preprocess_pipeline(self, df: pd.DataFrame, demographic_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input crime dataframe
            demographic_df (Optional[pd.DataFrame]): Demographic dataframe
            
        Returns:
            pd.DataFrame: Fully processed dataframe
        """
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Clean missing values
        df_processed = self.clean_missing_values(df)
        
        # Step 2: Remove outliers from numerical columns
        numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_columns:
            df_processed = self.remove_outliers(df_processed, numerical_columns)
        
        # Step 3: Extract temporal features
        if 'date' in df_processed.columns:
            df_processed = self.extract_temporal_features(df_processed)
        
        # Step 4: Merge with demographic data if provided
        if demographic_df is not None:
            df_processed = self.merge_demographic_data(df_processed, demographic_df)
        
        # Step 5: Encode categorical features
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            df_processed = self.encode_categorical_features(df_processed, categorical_columns)
        
        # Step 6: Create aggregated features
        df_processed = self.create_aggregated_features(df_processed)
        
        # Step 7: Normalize numerical features
        numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID columns and encoded columns from normalization
        numerical_columns = [col for col in numerical_columns 
                           if not col.endswith('_encoded') and 'id' not in col.lower()]
        if numerical_columns:
            df_processed = self.normalize_features(df_processed, numerical_columns)
        
        self.processed_data = df_processed
        logger.info(f"Preprocessing completed. Final shape: {df_processed.shape}")
        
        return df_processed
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to file.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            filename (str): Output filename
        """
        output_path = self.config.get('data', {}).get('processed_data_path', 'data/processed/')
        ensure_dir_exists(output_path)
        
        full_path = f"{output_path}/{filename}"
        df.to_csv(full_path, index=False)
        logger.info(f"Processed data saved to {full_path}")
    
    def get_preprocessing_summary(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict:
        """
        Get summary of preprocessing steps.
        
        Args:
            original_df (pd.DataFrame): Original dataframe
            processed_df (pd.DataFrame): Processed dataframe
            
        Returns:
            Dict: Summary statistics
        """
        summary = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'features_added': processed_df.shape[1] - original_df.shape[1],
            'rows_removed': original_df.shape[0] - processed_df.shape[0],
            'missing_values_original': original_df.isnull().sum().sum(),
            'missing_values_processed': processed_df.isnull().sum().sum(),
            'categorical_columns': len(processed_df.select_dtypes(include=['object']).columns),
            'numerical_columns': len(processed_df.select_dtypes(include=[np.number]).columns)
        }
        
        return summary

def main():
    """
    Main function to demonstrate preprocessing capabilities.
    """
    from src.utils import create_sample_crime_data, create_sample_demographic_data
    
    # Create sample data
    crime_data = create_sample_crime_data(1000)
    demographic_data = create_sample_demographic_data()
    
    # Initialize preprocessor
    preprocessor = CrimeDataPreprocessor()
    
    # Run preprocessing pipeline
    processed_data = preprocessor.preprocess_pipeline(crime_data, demographic_data)
    
    # Get summary
    summary = preprocessor.get_preprocessing_summary(crime_data, processed_data)
    print("\nPreprocessing Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save processed data
    preprocessor.save_processed_data(processed_data, "crime_data_processed.csv")
    
    print(f"\nProcessed data columns ({len(processed_data.columns)}):")
    print(processed_data.columns.tolist())

if __name__ == "__main__":
    main()