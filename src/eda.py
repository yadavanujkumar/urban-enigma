"""
Exploratory Data Analysis module for crime hotspot prediction.
Provides comprehensive analysis and visualization of crime data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, ensure_dir_exists
import logging

logger = logging.getLogger(__name__)

class CrimeDataAnalyzer:
    """
    Comprehensive EDA class for crime data analysis and visualization.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = load_config(config_path) if os.path.exists(config_path) else {}
        self.df = None
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load processed crime data.
        
        Args:
            file_path (str): Path to processed data file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def get_data_overview(self, df: pd.DataFrame = None) -> dict:
        """
        Get basic overview of the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            
        Returns:
            dict: Overview statistics
        """
        if df is None:
            df = self.df
            
        overview = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return overview
    
    def plot_crime_trends_by_year(self, df: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Plot year-wise crime trends.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            save_path (str): Path to save the plot
        """
        if df is None:
            df = self.df
            
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Crime count by year
        plt.subplot(1, 2, 1)
        yearly_crimes = df.groupby('year').size()
        yearly_crimes.plot(kind='line', marker='o', linewidth=2, markersize=6)
        plt.title('Crime Trends by Year', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Crimes')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Crime rate growth
        plt.subplot(1, 2, 2)
        crime_growth = yearly_crimes.pct_change() * 100
        crime_growth.plot(kind='bar', color=['red' if x < 0 else 'green' for x in crime_growth])
        plt.title('Year-over-Year Crime Rate Change (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Growth Rate (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_crime_trends_by_state(self, df: pd.DataFrame = None, top_n: int = 10, save_path: str = None) -> None:
        """
        Plot state-wise crime trends.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            top_n (int): Number of top states to show
            save_path (str): Path to save the plot
        """
        if df is None:
            df = self.df
            
        # Get top states by crime count
        state_crimes = df.groupby('state').size().sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Bar chart of crime counts
        plt.subplot(2, 2, 1)
        state_crimes.plot(kind='bar', color='skyblue')
        plt.title(f'Top {top_n} States by Crime Count', fontsize=14, fontweight='bold')
        plt.xlabel('State')
        plt.ylabel('Number of Crimes')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Pie chart
        plt.subplot(2, 2, 2)
        plt.pie(state_crimes.values, labels=state_crimes.index, autopct='%1.1f%%', startangle=90)
        plt.title(f'Crime Distribution - Top {top_n} States', fontsize=14, fontweight='bold')
        
        # Plot 3: Crime types by top states
        plt.subplot(2, 1, 2)
        top_states = state_crimes.index[:5]
        crime_type_by_state = df[df['state'].isin(top_states)].groupby(['state', 'crime_type']).size().unstack(fill_value=0)
        crime_type_by_state.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Crime Types Distribution by Top 5 States', fontsize=14, fontweight='bold')
        plt.xlabel('State')
        plt.ylabel('Number of Crimes')
        plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_crime_intensity_heatmap(self, df: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Create heatmap of crime intensity by state and month.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            save_path (str): Path to save the plot
        """
        if df is None:
            df = self.df
            
        # Create pivot table for heatmap
        heatmap_data = df.groupby(['state', 'month']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, 
                    annot=True, 
                    fmt='d', 
                    cmap='YlOrRd', 
                    cbar_kws={'label': 'Number of Crimes'},
                    linewidths=0.5)
        plt.title('Crime Intensity Heatmap by State and Month', fontsize=16, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('State')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_crime_categories(self, df: pd.DataFrame = None, save_path: str = None) -> dict:
        """
        Analyze crime categories and their frequency.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            save_path (str): Path to save the plot
            
        Returns:
            dict: Crime category analysis results
        """
        if df is None:
            df = self.df
            
        crime_counts = df['crime_type'].value_counts()
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Bar chart
        plt.subplot(2, 2, 1)
        crime_counts.head(10).plot(kind='bar', color='lightcoral')
        plt.title('Top 10 Crime Categories', fontsize=14, fontweight='bold')
        plt.xlabel('Crime Type')
        plt.ylabel('Number of Crimes')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Pie chart for top categories
        plt.subplot(2, 2, 2)
        top_crimes = crime_counts.head(8)
        other_crimes = crime_counts[8:].sum()
        if other_crimes > 0:
            pie_data = list(top_crimes.values) + [other_crimes]
            pie_labels = list(top_crimes.index) + ['Others']
        else:
            pie_data = top_crimes.values
            pie_labels = top_crimes.index
            
        plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Crime Categories Distribution', fontsize=14, fontweight='bold')
        
        # Plot 3: Crime trends by category over time
        plt.subplot(2, 1, 2)
        top_5_crimes = crime_counts.head(5).index
        for crime in top_5_crimes:
            crime_by_year = df[df['crime_type'] == crime].groupby('year').size()
            plt.plot(crime_by_year.index, crime_by_year.values, marker='o', label=crime, linewidth=2)
        
        plt.title('Crime Trends by Category Over Years', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Crimes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Return analysis results
        analysis_results = {
            'total_categories': len(crime_counts),
            'most_common_crime': crime_counts.index[0],
            'most_common_count': crime_counts.iloc[0],
            'least_common_crime': crime_counts.index[-1],
            'least_common_count': crime_counts.iloc[-1],
            'crime_distribution': crime_counts.to_dict()
        }
        
        return analysis_results
    
    def plot_temporal_patterns(self, df: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Analyze temporal patterns in crime data.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            save_path (str): Path to save the plot
        """
        if df is None:
            df = self.df
            
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Monthly patterns
        plt.subplot(3, 2, 1)
        monthly_crimes = df.groupby('month').size()
        monthly_crimes.plot(kind='bar', color='steelblue')
        plt.title('Crime Patterns by Month', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Number of Crimes')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Weekday patterns
        plt.subplot(3, 2, 2)
        weekday_crimes = df.groupby('weekday_name').size()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_crimes = weekday_crimes.reindex(day_order)
        weekday_crimes.plot(kind='bar', color='orange')
        plt.title('Crime Patterns by Weekday', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Crimes')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Seasonal patterns
        plt.subplot(3, 2, 3)
        seasonal_crimes = df.groupby('season').size()
        seasonal_crimes.plot(kind='bar', color='green')
        plt.title('Crime Patterns by Season', fontsize=14, fontweight='bold')
        plt.xlabel('Season')
        plt.ylabel('Number of Crimes')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Holiday vs Non-holiday
        plt.subplot(3, 2, 4)
        holiday_crimes = df.groupby('is_holiday').size()
        holiday_labels = ['Non-Holiday', 'Holiday']
        plt.pie(holiday_crimes.values, labels=holiday_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Crimes on Holidays vs Non-Holidays', fontsize=14, fontweight='bold')
        
        # Plot 5: Time series of daily crimes
        plt.subplot(3, 1, 3)
        if 'date' in df.columns:
            daily_crimes = df.groupby('date').size()
            daily_crimes.plot(kind='line', alpha=0.7, color='purple')
            plt.title('Daily Crime Trends Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Number of Crimes')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_demographic_correlations(self, df: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Analyze correlations between crime and demographic factors.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            save_path (str): Path to save the plot
        """
        if df is None:
            df = self.df
            
        # Select demographic and crime-related columns
        demo_cols = ['population', 'literacy_rate', 'unemployment_rate', 'urban_population_pct', 
                     'police_stations', 'area_sq_km', 'gdp_per_capita']
        crime_cols = ['state_crime_count', 'latitude', 'longitude']
        
        correlation_cols = [col for col in demo_cols + crime_cols if col in df.columns]
        
        if len(correlation_cols) > 1:
            plt.figure(figsize=(12, 10))
            
            # Correlation heatmap
            correlation_matrix = df[correlation_cols].corr()
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True,
                       fmt='.2f')
            plt.title('Correlation Matrix: Crime vs Demographic Factors', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_interactive_crime_map(self, df: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Create interactive crime map using Plotly.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            save_path (str): Path to save the plot
        """
        if df is None:
            df = self.df
            
        # Sample data for better visualization
        if len(df) > 1000:
            df_sample = df.sample(1000)
        else:
            df_sample = df
            
        fig = px.scatter_mapbox(df_sample,
                               lat="latitude",
                               lon="longitude",
                               color="crime_type",
                               size="state_crime_count",
                               hover_name="state",
                               hover_data=["district", "year", "month"],
                               title="Crime Hotspots Across India",
                               mapbox_style="open-street-map",
                               zoom=4,
                               center=dict(lat=20.5937, lon=78.9629),  # India center
                               height=600)
        
        fig.update_layout(
            title_font_size=20,
            title_x=0.5,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        fig.show()
        
        if save_path:
            fig.write_html(save_path)
    
    def generate_comprehensive_report(self, df: pd.DataFrame = None, output_dir: str = "reports") -> dict:
        """
        Generate comprehensive EDA report.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            output_dir (str): Directory to save reports
            
        Returns:
            dict: Comprehensive analysis results
        """
        if df is None:
            df = self.df
            
        ensure_dir_exists(output_dir)
        
        # Generate all analyses
        overview = self.get_data_overview(df)
        crime_analysis = self.analyze_crime_categories(df, f"{output_dir}/crime_categories.png")
        
        # Generate all plots
        self.plot_crime_trends_by_year(df, f"{output_dir}/yearly_trends.png")
        self.plot_crime_trends_by_state(df, save_path=f"{output_dir}/state_trends.png")
        self.plot_crime_intensity_heatmap(df, f"{output_dir}/intensity_heatmap.png")
        self.plot_temporal_patterns(df, f"{output_dir}/temporal_patterns.png")
        self.plot_demographic_correlations(df, f"{output_dir}/demographic_correlations.png")
        self.create_interactive_crime_map(df, f"{output_dir}/interactive_map.html")
        
        # Compile comprehensive report
        report = {
            'data_overview': overview,
            'crime_analysis': crime_analysis,
            'key_insights': {
                'total_crimes': len(df),
                'unique_states': df['state'].nunique() if 'state' in df.columns else 0,
                'unique_districts': df['district'].nunique() if 'district' in df.columns else 0,
                'date_range': f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else "Unknown",
                'most_affected_state': df['state'].mode().iloc[0] if 'state' in df.columns else "Unknown",
                'peak_crime_month': df['month'].mode().iloc[0] if 'month' in df.columns else "Unknown"
            }
        }
        
        logger.info(f"Comprehensive EDA report generated in {output_dir}")
        return report

def main():
    """
    Main function to demonstrate EDA capabilities.
    """
    # Initialize analyzer
    analyzer = CrimeDataAnalyzer()
    
    # Load processed data
    data_path = "data/processed/crime_data_processed.csv"
    if os.path.exists(data_path):
        df = analyzer.load_data(data_path)
    else:
        # Create sample data if processed data doesn't exist
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.data_preprocessing import CrimeDataPreprocessor
        from src.utils import create_sample_crime_data, create_sample_demographic_data
        
        crime_data = create_sample_crime_data(1000)
        demographic_data = create_sample_demographic_data()
        
        preprocessor = CrimeDataPreprocessor()
        df = preprocessor.preprocess_pipeline(crime_data, demographic_data)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(df)
    
    print("\n" + "="*50)
    print("CRIME DATA ANALYSIS REPORT")
    print("="*50)
    
    print(f"\nData Overview:")
    print(f"  Total Records: {report['data_overview']['shape'][0]:,}")
    print(f"  Total Features: {report['data_overview']['shape'][1]:,}")
    print(f"  Memory Usage: {report['data_overview']['memory_usage_mb']:.2f} MB")
    
    print(f"\nKey Insights:")
    for key, value in report['key_insights'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nCrime Categories:")
    print(f"  Total Categories: {report['crime_analysis']['total_categories']}")
    print(f"  Most Common: {report['crime_analysis']['most_common_crime']} ({report['crime_analysis']['most_common_count']} cases)")
    print(f"  Least Common: {report['crime_analysis']['least_common_crime']} ({report['crime_analysis']['least_common_count']} cases)")
    
    print("\nEDA visualizations saved to 'reports/' directory")

if __name__ == "__main__":
    main()