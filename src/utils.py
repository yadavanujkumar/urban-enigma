"""
Utility functions for the crime hotspot prediction project.
"""

import pandas as pd
import numpy as np
import yaml
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import pickle
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def ensure_dir_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def save_model(model: Any, filepath: str, model_type: str = "pickle") -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath (str): Path to save the model
        model_type (str): Type of serialization ('pickle' or 'joblib')
    """
    ensure_dir_exists(os.path.dirname(filepath))
    
    try:
        if model_type == "joblib":
            joblib.dump(model, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def load_model(filepath: str, model_type: str = "pickle") -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        model_type (str): Type of serialization ('pickle' or 'joblib')
        
    Returns:
        Any: Loaded model object
    """
    try:
        if model_type == "joblib":
            model = joblib.load(filepath)
        else:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate if dataframe has required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): List of required column names
        
    Returns:
        bool: True if all required columns exist
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True

def get_sample_data_info() -> Dict[str, str]:
    """
    Get information about sample datasets available.
    
    Returns:
        Dict[str, str]: Dictionary with dataset names and descriptions
    """
    return {
        "crime_data_sample": "Sample crime records with location, type, and timestamp",
        "demographic_data_sample": "Sample demographic data by district/state",
        "india_states_geojson": "GeoJSON file with Indian state boundaries"
    }

def create_sample_crime_data(n_records: int = 1000) -> pd.DataFrame:
    """
    Create sample crime data for demonstration purposes.
    
    Args:
        n_records (int): Number of records to generate
        
    Returns:
        pd.DataFrame: Sample crime dataset
    """
    np.random.seed(42)
    
    # Indian states for sampling
    states = [
        'Maharashtra', 'Uttar Pradesh', 'Tamil Nadu', 'West Bengal', 'Karnataka',
        'Gujarat', 'Rajasthan', 'Andhra Pradesh', 'Odisha', 'Telangana',
        'Kerala', 'Madhya Pradesh', 'Punjab', 'Haryana', 'Bihar'
    ]
    
    # Crime types
    crime_types = [
        'Theft', 'Burglary', 'Assault', 'Robbery', 'Murder', 'Rape',
        'Kidnapping', 'Cybercrime', 'Drug Offenses', 'Fraud',
        'Domestic Violence', 'Vandalism'
    ]
    
    # Generate sample data
    data = {
        'incident_id': [f"INC_{i:06d}" for i in range(1, n_records + 1)],
        'state': np.random.choice(states, n_records),
        'district': [f"District_{i}" for i in np.random.randint(1, 50, n_records)],
        'crime_type': np.random.choice(crime_types, n_records),
        'year': np.random.randint(2018, 2024, n_records),
        'month': np.random.randint(1, 13, n_records),
        'day': np.random.randint(1, 29, n_records),
        'latitude': np.random.uniform(8.0, 37.0, n_records),  # India's latitude range
        'longitude': np.random.uniform(68.0, 97.0, n_records),  # India's longitude range
        'victim_age': np.random.randint(18, 80, n_records),
        'victim_gender': np.random.choice(['Male', 'Female'], n_records),
        'case_status': np.random.choice(['Open', 'Closed', 'Under Investigation'], n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Create datetime column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['weekday'] = df['date'].dt.day_name()
    
    logger.info(f"Created sample crime dataset with {n_records} records")
    return df

def create_sample_demographic_data() -> pd.DataFrame:
    """
    Create sample demographic data for Indian states.
    
    Returns:
        pd.DataFrame: Sample demographic dataset
    """
    states = [
        'Maharashtra', 'Uttar Pradesh', 'Tamil Nadu', 'West Bengal', 'Karnataka',
        'Gujarat', 'Rajasthan', 'Andhra Pradesh', 'Odisha', 'Telangana',
        'Kerala', 'Madhya Pradesh', 'Punjab', 'Haryana', 'Bihar'
    ]
    
    np.random.seed(42)
    
    data = {
        'state': states,
        'population': np.random.randint(10000000, 200000000, len(states)),
        'literacy_rate': np.random.uniform(60, 95, len(states)),
        'unemployment_rate': np.random.uniform(2, 15, len(states)),
        'urban_population_pct': np.random.uniform(20, 80, len(states)),
        'police_stations': np.random.randint(100, 2000, len(states)),
        'area_sq_km': np.random.randint(50000, 300000, len(states)),
        'gdp_per_capita': np.random.randint(50000, 500000, len(states))
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Created sample demographic dataset with {len(states)} states")
    return df

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
        
    Returns:
        float: Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

def get_indian_holidays(year: int) -> List[str]:
    """
    Get list of major Indian holidays for a given year.
    
    Args:
        year (int): Year for which to get holidays
        
    Returns:
        List[str]: List of holiday dates in YYYY-MM-DD format
    """
    # Major fixed holidays in India
    holidays = [
        f"{year}-01-26",  # Republic Day
        f"{year}-08-15",  # Independence Day
        f"{year}-10-02",  # Gandhi Jayanti
        f"{year}-12-25",  # Christmas
    ]
    
    # Note: In a real implementation, you would include variable holidays
    # like Diwali, Eid, Holi, etc. which change dates each year
    
    return holidays

def memory_usage(df: pd.DataFrame) -> str:
    """
    Get memory usage information for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        
    Returns:
        str: Memory usage summary
    """
    memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    return f"DataFrame memory usage: {memory_usage_mb:.2f} MB"