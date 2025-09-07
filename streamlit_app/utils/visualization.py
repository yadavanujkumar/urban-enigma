"""
Visualization utilities for Streamlit app.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
import logging

logger = logging.getLogger(__name__)

class CrimeVisualizer:
    """
    Visualization utility for crime data in Streamlit dashboard.
    """
    
    def __init__(self):
        """Initialize visualizer."""
        self.color_palette = px.colors.qualitative.Set3
        
    def create_crime_trend_chart(self, df: pd.DataFrame, time_col: str = 'year', 
                                value_col: str = None, title: str = "Crime Trends") -> go.Figure:
        """
        Create crime trend line chart.
        
        Args:
            df (pd.DataFrame): Input dataframe
            time_col (str): Time column name
            value_col (str): Value column (if None, will count records)
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if time_col not in df.columns:
                return go.Figure()
            
            if value_col is None:
                trend_data = df.groupby(time_col).size().reset_index(name='count')
                y_col = 'count'
                y_title = 'Crime Count'
            else:
                if value_col not in df.columns:
                    return go.Figure()
                trend_data = df.groupby(time_col)[value_col].sum().reset_index()
                y_col = value_col
                y_title = value_col.replace('_', ' ').title()
            
            fig = px.line(
                trend_data,
                x=time_col,
                y=y_col,
                title=title,
                labels={time_col: time_col.title(), y_col: y_title}
            )
            
            fig.update_traces(mode='lines+markers', line=dict(width=3))
            fig.update_layout(
                hovermode='x unified',
                xaxis_title=time_col.title(),
                yaxis_title=y_title
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating crime trend chart: {e}")
            return go.Figure()
    
    def create_geographic_distribution(self, df: pd.DataFrame, location_col: str = 'state',
                                     title: str = "Geographic Distribution") -> go.Figure:
        """
        Create geographic distribution chart.
        
        Args:
            df (pd.DataFrame): Input dataframe
            location_col (str): Location column name
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if location_col not in df.columns:
                return go.Figure()
            
            geo_data = df[location_col].value_counts().reset_index()
            geo_data.columns = [location_col, 'count']
            
            # Create bar chart for top locations
            fig = px.bar(
                geo_data.head(15),
                x='count',
                y=location_col,
                orientation='h',
                title=title,
                labels={'count': 'Crime Count', location_col: location_col.title()}
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating geographic distribution: {e}")
            return go.Figure()
    
    def create_crime_type_distribution(self, df: pd.DataFrame, crime_col: str = 'crime_type',
                                     chart_type: str = 'pie') -> go.Figure:
        """
        Create crime type distribution chart.
        
        Args:
            df (pd.DataFrame): Input dataframe
            crime_col (str): Crime type column name
            chart_type (str): Chart type ('pie', 'bar', 'treemap')
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if crime_col not in df.columns:
                return go.Figure()
            
            crime_data = df[crime_col].value_counts()
            
            if chart_type == 'pie':
                fig = px.pie(
                    values=crime_data.values,
                    names=crime_data.index,
                    title="Crime Type Distribution"
                )
            elif chart_type == 'bar':
                fig = px.bar(
                    x=crime_data.index,
                    y=crime_data.values,
                    title="Crime Type Distribution",
                    labels={'x': 'Crime Type', 'y': 'Count'}
                )
                fig.update_xaxis(tickangle=45)
            elif chart_type == 'treemap':
                fig = px.treemap(
                    names=crime_data.index,
                    values=crime_data.values,
                    title="Crime Type Distribution"
                )
            else:
                fig = go.Figure()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating crime type distribution: {e}")
            return go.Figure()
    
    def create_temporal_heatmap(self, df: pd.DataFrame, time_col1: str = 'month',
                              time_col2: str = 'weekday', title: str = "Temporal Crime Pattern") -> go.Figure:
        """
        Create temporal heatmap showing crime patterns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            time_col1 (str): First time dimension
            time_col2 (str): Second time dimension
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if time_col1 not in df.columns or time_col2 not in df.columns:
                return go.Figure()
            
            # Create pivot table
            heatmap_data = df.groupby([time_col1, time_col2]).size().unstack(fill_value=0)
            
            fig = px.imshow(
                heatmap_data,
                title=title,
                labels=dict(x=time_col2.title(), y=time_col1.title(), color="Crime Count"),
                aspect="auto"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating temporal heatmap: {e}")
            return go.Figure()
    
    def create_scatter_mapbox(self, df: pd.DataFrame, lat_col: str = 'latitude',
                            lon_col: str = 'longitude', color_col: str = None,
                            size_col: str = None, hover_cols: list = None) -> go.Figure:
        """
        Create scatter mapbox for crime locations.
        
        Args:
            df (pd.DataFrame): Input dataframe
            lat_col (str): Latitude column name
            lon_col (str): Longitude column name
            color_col (str): Column for color coding
            size_col (str): Column for size coding
            hover_cols (list): Columns to show on hover
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if lat_col not in df.columns or lon_col not in df.columns:
                return go.Figure()
            
            # Sample data if too large
            df_plot = df.sample(min(1000, len(df))) if len(df) > 1000 else df
            
            fig = px.scatter_mapbox(
                df_plot,
                lat=lat_col,
                lon=lon_col,
                color=color_col,
                size=size_col,
                hover_data=hover_cols,
                mapbox_style="open-street-map",
                zoom=4,
                center=dict(lat=20.5937, lon=78.9629),
                height=600,
                title="Crime Locations"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter mapbox: {e}")
            return go.Figure()
    
    def create_correlation_heatmap(self, df: pd.DataFrame, columns: list = None) -> go.Figure:
        """
        Create correlation heatmap for numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to include (if None, use all numerical)
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if columns is None:
                # Get numerical columns
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
                # Remove ID columns
                columns = [col for col in columns if 'id' not in col.lower()]
            
            if len(columns) < 2:
                return go.Figure()
            
            # Calculate correlation matrix
            corr_matrix = df[columns].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                labels=dict(color="Correlation"),
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            
            # Add correlation values as text
            fig.update_traces(
                text=corr_matrix.round(2),
                texttemplate="%{text}",
                textfont={"size": 10}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return go.Figure()
    
    def create_time_series_plot(self, ts: pd.Series, forecasts: dict = None,
                              title: str = "Time Series") -> go.Figure:
        """
        Create time series plot with optional forecasts.
        
        Args:
            ts (pd.Series): Time series data
            forecasts (dict): Dictionary of forecast series
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=ts.index,
                y=ts.values,
                mode='lines',
                name='Historical Data',
                line=dict(color='blue', width=2)
            ))
            
            # Add forecasts if provided
            if forecasts:
                colors = ['red', 'green', 'purple', 'orange']
                for i, (model_name, forecast_series) in enumerate(forecasts.items()):
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=forecast_series.index,
                        y=forecast_series.values,
                        mode='lines',
                        name=f'{model_name} Forecast',
                        line=dict(color=color, width=2, dash='dash')
                    ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Crime Count",
                hovermode='x unified',
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            return go.Figure()
    
    def create_feature_importance_chart(self, importance_dict: dict, top_n: int = 20,
                                      title: str = "Feature Importance") -> go.Figure:
        """
        Create feature importance bar chart.
        
        Args:
            importance_dict (dict): Feature importance dictionary
            top_n (int): Number of top features to show
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if not importance_dict:
                return go.Figure()
            
            # Sort by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_n]
            
            features, importances = zip(*top_features)
            
            fig = px.bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                title=title,
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, len(features) * 25)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance chart: {e}")
            return go.Figure()
    
    def create_prediction_confidence_chart(self, predictions: dict, title: str = "Prediction Confidence") -> go.Figure:
        """
        Create prediction confidence chart.
        
        Args:
            predictions (dict): Prediction probabilities
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if not predictions:
                return go.Figure()
            
            crime_types = list(predictions.keys())
            probabilities = list(predictions.values())
            
            fig = px.bar(
                x=probabilities,
                y=crime_types,
                orientation='h',
                title=title,
                labels={'x': 'Probability', 'y': 'Crime Type'}
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(300, len(crime_types) * 30)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating prediction confidence chart: {e}")
            return go.Figure()
    
    def create_summary_dashboard(self, df: pd.DataFrame) -> dict:
        """
        Create summary dashboard with multiple charts.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Dictionary of figures
        """
        figures = {}
        
        try:
            # Crime trend over time
            if 'year' in df.columns:
                figures['trend'] = self.create_crime_trend_chart(df, 'year')
            
            # Geographic distribution
            if 'state' in df.columns:
                figures['geographic'] = self.create_geographic_distribution(df, 'state')
            
            # Crime type distribution
            if 'crime_type' in df.columns:
                figures['crime_types'] = self.create_crime_type_distribution(df)
            
            # Temporal patterns
            if 'month' in df.columns and 'weekday' in df.columns:
                figures['temporal'] = self.create_temporal_heatmap(df)
            
            # Crime map
            if 'latitude' in df.columns and 'longitude' in df.columns:
                figures['map'] = self.create_scatter_mapbox(df)
            
        except Exception as e:
            logger.error(f"Error creating summary dashboard: {e}")
        
        return figures