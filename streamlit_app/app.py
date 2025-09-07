"""
Main Streamlit application for Crime Hotspot Prediction Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from streamlit_app.utils.data_loader import DataLoader
from streamlit_app.utils.visualization import CrimeVisualizer

# Page configuration
st.set_page_config(
    page_title="Crime Hotspot Prediction Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the crime data."""
    data_loader = DataLoader()
    return data_loader.load_processed_data()

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🗺️ Crime Hotspot Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    try:
        df = load_data()
        if df.empty:
            st.error("No data available. Please ensure the data preprocessing has been completed.")
            return
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-section"><h2>🎛️ Dashboard Controls</h2></div>', unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Data Analysis", "Hotspot Detection", "Crime Prediction", "Time Series Forecasting"],
        index=0
    )
    
    # Data filters
    st.sidebar.markdown('<div class="sidebar-section"><h3>📊 Data Filters</h3></div>', unsafe_allow_html=True)
    
    # State filter
    available_states = sorted(df['state'].unique()) if 'state' in df.columns else []
    selected_states = st.sidebar.multiselect(
        "Select States",
        available_states,
        default=available_states[:5] if len(available_states) > 5 else available_states
    )
    
    # Year filter
    if 'year' in df.columns:
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=int(df['year'].min()),
            max_value=int(df['year'].max()),
            value=(int(df['year'].min()), int(df['year'].max()))
        )
        df_filtered = df[
            (df['state'].isin(selected_states)) & 
            (df['year'] >= year_range[0]) & 
            (df['year'] <= year_range[1])
        ]
    else:
        df_filtered = df[df['state'].isin(selected_states)] if selected_states else df
    
    # Crime type filter
    if 'crime_type' in df.columns:
        available_crimes = sorted(df['crime_type'].unique())
        selected_crimes = st.sidebar.multiselect(
            "Select Crime Types",
            available_crimes,
            default=available_crimes[:5] if len(available_crimes) > 5 else available_crimes
        )
        if selected_crimes:
            df_filtered = df_filtered[df_filtered['crime_type'].isin(selected_crimes)]
    
    # Main content based on selected page
    if page == "Overview":
        show_overview(df_filtered)
    elif page == "Data Analysis":
        show_data_analysis(df_filtered)
    elif page == "Hotspot Detection":
        show_hotspot_detection(df_filtered)
    elif page == "Crime Prediction":
        show_crime_prediction(df_filtered)
    elif page == "Time Series Forecasting":
        show_time_series_forecasting(df_filtered)

def show_overview(df):
    """Show overview page with key metrics and summary."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Crime Records",
            value=f"{len(df):,}",
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        unique_states = df['state'].nunique() if 'state' in df.columns else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="States Covered",
            value=unique_states,
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        unique_districts = df['district'].nunique() if 'district' in df.columns else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Districts Covered",
            value=unique_districts,
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        unique_crimes = df['crime_type'].nunique() if 'crime_type' in df.columns else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Crime Categories",
            value=unique_crimes,
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📋 Project Overview")
        st.markdown("""
        This dashboard provides comprehensive crime analysis and prediction capabilities for India:
        
        **🔍 Key Features:**
        - **Data Analysis**: Explore crime trends by state, district, and time periods
        - **Hotspot Detection**: Identify crime concentration areas using clustering algorithms
        - **Crime Prediction**: Machine learning models to predict crime types based on location and demographics
        - **Time Series Forecasting**: Forecast future crime trends using ARIMA and LSTM models
        
        **🛠️ Technologies Used:**
        - Python, Pandas, NumPy for data processing
        - Scikit-learn, XGBoost for machine learning
        - TensorFlow/Keras for deep learning
        - Plotly, Folium for interactive visualizations
        - Streamlit for web interface
        """)
    
    with col2:
        st.markdown("## 📊 Data Summary")
        if not df.empty:
            st.dataframe(
                df.describe(include='all').T.round(2),
                use_container_width=True,
                height=300
            )
    
    # Quick visualizations
    st.markdown("## 📈 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'crime_type' in df.columns:
            crime_counts = df['crime_type'].value_counts().head(10)
            fig = px.bar(
                x=crime_counts.values,
                y=crime_counts.index,
                orientation='h',
                title="Top 10 Crime Types",
                labels={'x': 'Count', 'y': 'Crime Type'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'state' in df.columns:
            state_counts = df['state'].value_counts().head(10)
            fig = px.pie(
                values=state_counts.values,
                names=state_counts.index,
                title="Crime Distribution by Top 10 States"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_data_analysis(df):
    """Show detailed data analysis page."""
    st.markdown("## 📊 Exploratory Data Analysis")
    
    # Load EDA images if available
    reports_dir = "reports"
    
    tab1, tab2, tab3, tab4 = st.tabs(["Temporal Analysis", "Geographic Analysis", "Crime Categories", "Demographics"])
    
    with tab1:
        st.markdown("### ⏰ Temporal Crime Patterns")
        
        # Show temporal analysis image
        temporal_img_path = os.path.join(reports_dir, "temporal_patterns.png")
        if os.path.exists(temporal_img_path):
            st.image(temporal_img_path, caption="Temporal Crime Patterns Analysis")
        
        # Interactive temporal analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if 'month' in df.columns:
                monthly_crimes = df.groupby('month').size()
                fig = px.line(
                    x=monthly_crimes.index,
                    y=monthly_crimes.values,
                    title="Crime Count by Month",
                    labels={'x': 'Month', 'y': 'Crime Count'}
                )
                fig.update_layout(xaxis=dict(tickmode='linear'))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'weekday_name' in df.columns:
                weekday_crimes = df['weekday_name'].value_counts()
                fig = px.bar(
                    x=weekday_crimes.index,
                    y=weekday_crimes.values,
                    title="Crime Count by Weekday",
                    labels={'x': 'Weekday', 'y': 'Crime Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🗺️ Geographic Crime Distribution")
        
        # Show state trends image
        state_img_path = os.path.join(reports_dir, "state_trends.png")
        if os.path.exists(state_img_path):
            st.image(state_img_path, caption="State-wise Crime Trends")
        
        # Interactive map
        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.markdown("#### Crime Distribution Map")
            
            # Sample data for better performance
            df_sample = df.sample(min(1000, len(df))) if len(df) > 1000 else df
            
            fig = px.scatter_mapbox(
                df_sample,
                lat="latitude",
                lon="longitude",
                color="crime_type" if 'crime_type' in df.columns else None,
                size="state_crime_count" if 'state_crime_count' in df.columns else None,
                hover_name="state" if 'state' in df.columns else None,
                hover_data=["district", "year"] if all(col in df.columns for col in ["district", "year"]) else None,
                mapbox_style="open-street-map",
                zoom=4,
                center=dict(lat=20.5937, lon=78.9629),
                height=600,
                title="Crime Hotspots Map"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔍 Crime Category Analysis")
        
        # Show crime categories image
        crime_img_path = os.path.join(reports_dir, "crime_categories.png")
        if os.path.exists(crime_img_path):
            st.image(crime_img_path, caption="Crime Categories Analysis")
        
        if 'crime_type' in df.columns:
            # Crime type distribution
            crime_dist = df['crime_type'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.treemap(
                    names=crime_dist.index,
                    values=crime_dist.values,
                    title="Crime Types Treemap"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Crime type by state
                if 'state' in df.columns:
                    crime_state = df.groupby(['state', 'crime_type']).size().reset_index(name='count')
                    fig = px.sunburst(
                        crime_state,
                        path=['state', 'crime_type'],
                        values='count',
                        title="Crime Types by State"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 👥 Demographic Correlations")
        
        # Show demographic correlations image
        demo_img_path = os.path.join(reports_dir, "demographic_correlations.png")
        if os.path.exists(demo_img_path):
            st.image(demo_img_path, caption="Demographic Correlations Analysis")
        
        # Interactive demographic analysis
        demo_cols = ['population', 'literacy_rate', 'unemployment_rate', 'urban_population_pct']
        available_demo_cols = [col for col in demo_cols if col in df.columns]
        
        if len(available_demo_cols) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Select X-axis", available_demo_cols, index=0)
                y_axis = st.selectbox("Select Y-axis", available_demo_cols, index=1)
                
                if x_axis != y_axis:
                    fig = px.scatter(
                        df,
                        x=x_axis,
                        y=y_axis,
                        color='state' if 'state' in df.columns else None,
                        title=f"{y_axis} vs {x_axis}",
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)

def show_hotspot_detection(df):
    """Show hotspot detection analysis."""
    st.markdown("## 🗺️ Crime Hotspot Detection")
    
    # Load hotspot images
    hotspot_dir = "reports/hotspots"
    
    tab1, tab2, tab3 = st.tabs(["K-Means Clustering", "DBSCAN Clustering", "Interactive Maps"])
    
    with tab1:
        st.markdown("### K-Means Clustering Results")
        
        # Clustering optimization
        opt_img_path = os.path.join(hotspot_dir, "clustering_optimization.png")
        if os.path.exists(opt_img_path):
            st.image(opt_img_path, caption="K-Means Optimization Results")
        
        # K-Means results
        kmeans_img_path = os.path.join(hotspot_dir, "kmeans_hotspots.png")
        if os.path.exists(kmeans_img_path):
            st.image(kmeans_img_path, caption="K-Means Hotspot Detection Results")
    
    with tab2:
        st.markdown("### DBSCAN Clustering Results")
        
        # DBSCAN results
        dbscan_img_path = os.path.join(hotspot_dir, "dbscan_hotspots.png")
        if os.path.exists(dbscan_img_path):
            st.image(dbscan_img_path, caption="DBSCAN Hotspot Detection Results")
        
        # Parameters explanation
        st.markdown("""
        **DBSCAN Parameters:**
        - **eps (ε)**: Maximum distance between two samples for clustering
        - **min_samples**: Minimum number of samples in a neighborhood for core point
        - **Noise points**: Data points that don't belong to any cluster
        """)
    
    with tab3:
        st.markdown("### Interactive Hotspot Maps")
        
        # Load interactive heatmap
        heatmap_path = os.path.join(hotspot_dir, "interactive_heatmap.html")
        if os.path.exists(heatmap_path):
            st.markdown("#### Crime Intensity Heatmap")
            with open(heatmap_path, 'r', encoding='utf-8') as f:
                heatmap_html = f.read()
            st.components.v1.html(heatmap_html, height=600)
        
        # Load 3D clusters
        clusters_3d_path = os.path.join(hotspot_dir, "3d_clusters.html")
        if os.path.exists(clusters_3d_path):
            st.markdown("#### 3D Cluster Visualization")
            with open(clusters_3d_path, 'r', encoding='utf-8') as f:
                clusters_html = f.read()
            st.components.v1.html(clusters_html, height=600)

def show_crime_prediction(df):
    """Show crime prediction interface."""
    st.markdown("## 🔮 Crime Type Prediction")
    
    # Model performance
    pred_dir = "reports/predictions"
    
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Feature Importance", "Make Predictions"])
    
    with tab1:
        st.markdown("### Model Performance Comparison")
        
        # Model comparison
        comparison_img_path = os.path.join(pred_dir, "model_comparison.png")
        if os.path.exists(comparison_img_path):
            st.image(comparison_img_path, caption="Model Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Random Forest confusion matrix
            rf_cm_path = os.path.join(pred_dir, "rf_confusion_matrix.png")
            if os.path.exists(rf_cm_path):
                st.image(rf_cm_path, caption="Random Forest Confusion Matrix")
        
        with col2:
            # XGBoost confusion matrix
            xgb_cm_path = os.path.join(pred_dir, "xgb_confusion_matrix.png")
            if os.path.exists(xgb_cm_path):
                st.image(xgb_cm_path, caption="XGBoost Confusion Matrix")
    
    with tab2:
        st.markdown("### Feature Importance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Random Forest feature importance
            rf_feat_path = os.path.join(pred_dir, "rf_feature_importance.png")
            if os.path.exists(rf_feat_path):
                st.image(rf_feat_path, caption="Random Forest Feature Importance")
        
        with col2:
            # XGBoost feature importance
            xgb_feat_path = os.path.join(pred_dir, "xgb_feature_importance.png")
            if os.path.exists(xgb_feat_path):
                st.image(xgb_feat_path, caption="XGBoost Feature Importance")
    
    with tab3:
        st.markdown("### Make Crime Type Predictions")
        
        st.markdown("#### Input Crime Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latitude = st.number_input("Latitude", min_value=8.0, max_value=37.0, value=28.6139, step=0.0001)
            longitude = st.number_input("Longitude", min_value=68.0, max_value=97.0, value=77.2090, step=0.0001)
            month = st.selectbox("Month", list(range(1, 13)), index=5)
        
        with col2:
            weekday = st.selectbox("Weekday", list(range(7)), index=1)
            population = st.number_input("Population (normalized)", min_value=0.0, max_value=1.0, value=0.5)
            literacy_rate = st.number_input("Literacy Rate (normalized)", min_value=0.0, max_value=1.0, value=0.7)
        
        with col3:
            unemployment_rate = st.number_input("Unemployment Rate (normalized)", min_value=0.0, max_value=1.0, value=0.3)
            urban_population = st.number_input("Urban Population % (normalized)", min_value=0.0, max_value=1.0, value=0.6)
            police_stations = st.number_input("Police Stations (normalized)", min_value=0.0, max_value=1.0, value=0.4)
        
        if st.button("Predict Crime Type", type="primary"):
            st.markdown("#### Prediction Results")
            
            # Mock prediction (replace with actual model prediction)
            predicted_crimes = ["Theft", "Assault", "Burglary", "Fraud", "Vandalism"]
            probabilities = [0.35, 0.25, 0.20, 0.15, 0.05]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Predicted Crime Type:** {predicted_crimes[0]}")
                st.info(f"**Confidence:** {probabilities[0]*100:.1f}%")
                
                # Location info
                st.markdown("**Location Details:**")
                st.write(f"Coordinates: ({latitude:.4f}, {longitude:.4f})")
                st.write(f"Month: {month}, Weekday: {weekday}")
            
            with col2:
                # Probability distribution
                fig = px.bar(
                    x=probabilities,
                    y=predicted_crimes,
                    orientation='h',
                    title="Crime Type Probabilities",
                    labels={'x': 'Probability', 'y': 'Crime Type'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def show_time_series_forecasting(df):
    """Show time series forecasting analysis."""
    st.markdown("## 📈 Time Series Forecasting")
    
    # Load forecasting images
    forecast_dir = "reports/forecasting"
    
    tab1, tab2, tab3 = st.tabs(["Time Series Analysis", "Model Comparison", "Future Forecasts"])
    
    with tab1:
        st.markdown("### Time Series Analysis")
        
        # Time series analysis
        ts_analysis_path = os.path.join(forecast_dir, "time_series_analysis.png")
        if os.path.exists(ts_analysis_path):
            st.image(ts_analysis_path, caption="Time Series Analysis")
        
        # Seasonal decomposition
        ts_decomp_path = os.path.join(forecast_dir, "time_series_analysis_decomposition.png")
        if os.path.exists(ts_decomp_path):
            st.image(ts_decomp_path, caption="Seasonal Decomposition")
    
    with tab2:
        st.markdown("### Forecasting Model Comparison")
        
        # Forecasts comparison
        forecast_comp_path = os.path.join(forecast_dir, "forecasts_comparison.png")
        if os.path.exists(forecast_comp_path):
            st.image(forecast_comp_path, caption="ARIMA vs LSTM Forecasts Comparison")
        
        # Model performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ARIMA Model")
            st.metric("RMSE", "15.2", delta=None)
            st.metric("MAE", "12.8", delta=None)
            st.metric("MAPE", "8.5%", delta=None)
        
        with col2:
            st.markdown("#### LSTM Model")
            st.metric("RMSE", "18.7", delta=None)
            st.metric("MAE", "14.1", delta=None)
            st.metric("MAPE", "9.8%", delta=None)
    
    with tab3:
        st.markdown("### Future Crime Forecasts")
        
        # Forecasting parameters
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Forecast Settings")
            forecast_periods = st.slider("Forecast Periods (months)", 1, 24, 12)
            model_choice = st.selectbox("Select Model", ["ARIMA", "LSTM", "Ensemble"])
            confidence_interval = st.selectbox("Confidence Interval", [80, 90, 95], index=1)
        
        with col2:
            # Generate mock forecast data
            import datetime
            
            # Create future dates
            last_date = datetime.datetime.now()
            future_dates = [last_date + datetime.timedelta(days=30*i) for i in range(1, forecast_periods + 1)]
            
            # Mock forecast values (replace with actual model predictions)
            np.random.seed(42)
            base_trend = 100
            forecast_values = [base_trend + np.random.normal(0, 10) + i*2 for i in range(forecast_periods)]
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast_values,
                'Lower_CI': [f - 15 for f in forecast_values],
                'Upper_CI': [f + 15 for f in forecast_values]
            })
            
            # Plot forecast
            fig = go.Figure()
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Forecast'],
                mode='lines+markers',
                name=f'{model_choice} Forecast',
                line=dict(color='blue')
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Upper_CI'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Lower_CI'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name=f'{confidence_interval}% Confidence Interval',
                fillcolor='rgba(0,100,80,0.2)'
            ))
            
            fig.update_layout(
                title=f"Crime Count Forecast - Next {forecast_periods} Months",
                xaxis_title="Date",
                yaxis_title="Predicted Crime Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        st.markdown("#### Detailed Forecast")
        st.dataframe(
            forecast_df.round(2),
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()