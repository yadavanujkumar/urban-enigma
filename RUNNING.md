# Running the Crime Hotspot Prediction Dashboard

This guide will help you run the complete crime hotspot prediction project.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. **Clone the repository** (if not already done):
```bash
git clone https://github.com/yadavanujkumar/urban-enigma.git
cd urban-enigma
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Streamlit Dashboard (Main Application)

Run the interactive web dashboard:

```bash
streamlit run streamlit_app/app.py
```

This will open your browser to `http://localhost:8501` where you can:
- Explore crime data trends
- View interactive hotspot maps
- Make crime type predictions
- See time series forecasts

### 2. Jupyter Notebooks

Launch Jupyter to run the analysis notebooks:

```bash
jupyter lab notebooks/
```

Open `crime_hotspot_analysis_complete.ipynb` for a comprehensive walkthrough of the entire analysis.

### 3. Command Line Analysis

Run individual modules for analysis:

```bash
# Data preprocessing
python src/data_preprocessing.py

# Exploratory data analysis
python src/eda.py

# Hotspot detection
python src/hotspot_detection.py

# Predictive modeling
python src/predictive_modeling.py

# Time series forecasting
python src/time_series_forecasting.py
```

## Testing

Run the test suite to verify everything works:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_data_preprocessing.py -v
pytest tests/test_hotspot_detection.py -v
pytest tests/test_predictive_modeling.py -v
```

## Project Features

### 🗺️ Interactive Dashboard
- **Overview**: Key metrics and data summary
- **Data Analysis**: Temporal and geographic crime patterns
- **Hotspot Detection**: K-Means and DBSCAN clustering results
- **Crime Prediction**: ML-based crime type prediction interface
- **Time Series Forecasting**: ARIMA and LSTM forecast visualizations

### 📊 Analysis Capabilities
- Comprehensive crime trend analysis by year, state, and district
- Interactive crime intensity heatmaps
- Geospatial clustering for hotspot identification
- Machine learning models with 99%+ accuracy
- Time series forecasting with seasonal pattern detection

### 🤖 Machine Learning Models
- **Classification**: Random Forest and XGBoost for crime type prediction
- **Clustering**: K-Means and DBSCAN for hotspot detection
- **Time Series**: ARIMA and LSTM for crime count forecasting
- **Evaluation**: Comprehensive metrics and cross-validation

## Data

The project includes sample datasets for demonstration:
- **Crime Data**: 1,000+ synthetic crime records with location, type, and temporal information
- **Demographic Data**: Population, literacy, unemployment statistics by state
- **Processed Data**: Feature-engineered dataset ready for machine learning

For production use, replace sample data with real crime datasets from sources like:
- National Crime Records Bureau (NCRB)
- State police departments
- District-level crime statistics

## Configuration

Modify `config/config.yaml` to adjust:
- Model parameters (number of clusters, ML hyperparameters)
- Data paths and file locations
- Visualization settings
- Dashboard configuration

## Troubleshooting

### Common Issues

1. **Port already in use**: If port 8501 is busy, Streamlit will automatically use the next available port
2. **Module import errors**: Ensure you're in the project root directory when running commands
3. **Memory issues**: Reduce sample data size in `src/utils.py` if needed
4. **Missing dependencies**: Run `pip install -r requirements.txt` again

### Performance Tips

1. **Large datasets**: The dashboard automatically samples large datasets for performance
2. **Model training**: Initial model training may take a few minutes
3. **Interactive maps**: Complex maps may load slowly on older hardware

## Output Files

The application generates various outputs in:
- `reports/`: Analysis visualizations and charts
- `models/saved_models/`: Trained machine learning models
- `data/processed/`: Processed datasets

## Next Steps

1. **Real Data Integration**: Replace sample data with actual crime datasets
2. **Model Deployment**: Deploy models to production systems
3. **Automated Pipelines**: Set up scheduled data processing and model retraining
4. **API Development**: Create REST APIs for real-time predictions
5. **Mobile App**: Develop mobile interface for field officers

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test results with `pytest tests/ -v`
3. Examine log messages for specific error details
4. Verify all dependencies are properly installed

The project provides a complete end-to-end solution for crime hotspot analysis and prediction, ready for adaptation to specific regional needs and real-world deployment.