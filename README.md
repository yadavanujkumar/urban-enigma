# Crime Hotspot Prediction in India - Urban Enigma

A comprehensive data science project that predicts and visualizes crime hotspots in India using publicly available datasets such as NCRB reports and Crime in India data.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for crime analysis and prediction, featuring:

- **Data Preprocessing**: Clean and transform crime datasets with temporal and demographic features
- **Exploratory Data Analysis**: Interactive visualizations of crime trends across Indian states and districts
- **Hotspot Detection**: Clustering algorithms to identify crime concentration areas
- **Predictive Modeling**: Machine learning models to predict crime types and future occurrences
- **Interactive Dashboard**: Streamlit-based web application with geospatial visualizations

## 📁 Project Structure

```
urban-enigma/
├── data/                          # Data storage
│   ├── raw/                       # Original datasets
│   ├── processed/                 # Cleaned and processed data
│   └── sample_data/               # Sample/demo datasets
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_hotspot_detection.ipynb
│   ├── 04_predictive_modeling.ipynb
│   └── 05_time_series_forecasting.ipynb
├── src/                           # Core Python modules
│   ├── data_preprocessing.py      # Data cleaning and transformation
│   ├── eda.py                     # Exploratory data analysis
│   ├── hotspot_detection.py       # Clustering and hotspot identification
│   ├── predictive_modeling.py     # Classification models
│   ├── time_series_forecasting.py # Time series analysis
│   └── utils.py                   # Utility functions
├── streamlit_app/                 # Interactive dashboard
│   ├── app.py                     # Main Streamlit application
│   ├── pages/                     # Dashboard pages
│   └── utils/                     # Dashboard utilities
├── models/                        # Saved ML models
├── tests/                         # Unit tests
└── config/                        # Configuration files
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/urban-enigma.git
cd urban-enigma
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. **Launch Jupyter Notebooks** for data analysis:
```bash
jupyter lab notebooks/
```

2. **Run the Streamlit Dashboard**:
```bash
streamlit run streamlit_app/app.py
```

3. **Execute individual modules**:
```bash
python src/data_preprocessing.py
python src/eda.py
python src/hotspot_detection.py
```

## 📊 Features

### Data Preprocessing
- ✅ Missing value imputation and outlier detection
- ✅ Categorical encoding for crime types, states, and districts
- ✅ Temporal feature extraction (year, month, weekday, holidays)
- ✅ Demographic data integration (population, literacy, workforce)

### Exploratory Data Analysis
- 📈 Year-wise and state-wise crime trend analysis
- 🗺️ Crime intensity heatmaps by geographical regions
- 📊 Crime category frequency analysis
- 🎯 Interactive visualizations with Plotly and Seaborn

### Hotspot Detection
- 🔍 K-Means and DBSCAN clustering for crime concentration
- 🗺️ Geospatial heatmaps with Folium
- 📍 Interactive maps showing crime hotspots across India

### Predictive Modeling
- 🤖 Random Forest and XGBoost for crime type classification
- 📈 ARIMA and LSTM models for time series forecasting
- 📊 Comprehensive model evaluation and metrics

### Interactive Dashboard
- 🌐 Streamlit-based web interface
- 🗺️ India map with color-coded crime hotspots
- 📊 Real-time trend analysis charts
- 🔮 Predictive model results and forecasts

## 📈 Model Performance

- **Classification Models**: Accuracy, Precision, Recall, F1-Score
- **Clustering**: Silhouette Score, Calinski-Harabasz Index
- **Time Series**: RMSE, MAE, MAPE

## 🛠️ Technologies Used

- **Data Science**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly, Folium
- **Machine Learning**: XGBoost, TensorFlow, Statsmodels
- **Web Framework**: Streamlit
- **Geospatial**: GeoPandas, Shapely
- **Development**: Jupyter, Pytest

## 📄 Data Sources

This project works with publicly available crime datasets including:
- National Crime Records Bureau (NCRB) reports
- Crime in India annual reports
- District-level crime statistics
- Demographic and socio-economic indicators

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- National Crime Records Bureau (NCRB) for providing crime data
- Open source community for the amazing tools and libraries
- Contributors and researchers in crime analytics and data science