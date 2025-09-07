"""
Time series forecasting module for crime prediction.
Implements ARIMA and LSTM models for crime count forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, ensure_dir_exists, save_model, load_model
import logging

logger = logging.getLogger(__name__)

class CrimeTimeSeriesForecaster:
    """
    Time series forecasting for crime data using ARIMA and LSTM models.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize time series forecaster with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = load_config(config_path) if os.path.exists(config_path) else {}
        self.df = None
        self.time_series = None
        self.models = {}
        self.scalers = {}
        self.forecasts = {}
        
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
    
    def prepare_time_series(self, df: pd.DataFrame = None, freq: str = "M", 
                           agg_column: str = "incident_id") -> pd.Series:
        """
        Prepare time series data for forecasting.
        
        Args:
            df (pd.DataFrame): Input dataframe
            freq (str): Time frequency ('D', 'W', 'M', 'Q', 'Y')
            agg_column (str): Column to aggregate (count)
            
        Returns:
            pd.Series: Time series data
        """
        if df is None:
            df = self.df
            
        if 'date' not in df.columns:
            logger.error("Date column not found in dataframe")
            return pd.Series()
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Aggregate by time frequency
        if freq == "D":
            time_series = df.resample('D').size()
        elif freq == "W":
            time_series = df.resample('W').size()
        elif freq == "M":
            time_series = df.resample('M').size()
        elif freq == "Q":
            time_series = df.resample('Q').size()
        elif freq == "Y":
            time_series = df.resample('Y').size()
        else:
            logger.error(f"Unsupported frequency: {freq}")
            return pd.Series()
        
        # Fill missing values with 0
        time_series = time_series.fillna(0)
        
        self.time_series = time_series
        logger.info(f"Prepared time series with {len(time_series)} periods (frequency: {freq})")
        
        return time_series
    
    def analyze_time_series(self, ts: pd.Series = None, save_path: str = None) -> dict:
        """
        Analyze time series characteristics.
        
        Args:
            ts (pd.Series): Time series data
            save_path (str): Path to save analysis plots
            
        Returns:
            dict: Time series analysis results
        """
        if ts is None:
            ts = self.time_series
            
        if ts is None or len(ts) == 0:
            logger.error("No time series data available")
            return {}
        
        # Basic statistics
        analysis = {
            'length': len(ts),
            'start_date': str(ts.index[0]),
            'end_date': str(ts.index[-1]),
            'mean': ts.mean(),
            'std': ts.std(),
            'min': ts.min(),
            'max': ts.max(),
            'trend': 'increasing' if ts.iloc[-1] > ts.iloc[0] else 'decreasing'
        }
        
        # Plot time series analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original time series
        axes[0, 0].plot(ts.index, ts.values)
        axes[0, 0].set_title('Crime Count Time Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Crime Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution
        axes[0, 1].hist(ts.values, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Crime Count Distribution')
        axes[0, 1].set_xlabel('Crime Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling statistics
        rolling_mean = ts.rolling(window=12).mean()
        rolling_std = ts.rolling(window=12).std()
        
        axes[1, 0].plot(ts.index, ts.values, label='Original', alpha=0.7)
        axes[1, 0].plot(rolling_mean.index, rolling_mean.values, label='Rolling Mean', color='red')
        axes[1, 0].plot(rolling_std.index, rolling_std.values, label='Rolling Std', color='green')
        axes[1, 0].set_title('Rolling Statistics (12 periods)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Crime Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        try:
            plot_acf(ts.dropna(), lags=min(20, len(ts)//4), ax=axes[1, 1])
            axes[1, 1].set_title('Autocorrelation Function')
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'ACF plot failed: {str(e)}', 
                           transform=axes[1, 1].transAxes, ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Seasonal decomposition (if enough data)
        if len(ts) >= 24:
            try:
                decomposition = seasonal_decompose(ts, model='additive', period=12)
                
                fig, axes = plt.subplots(4, 1, figsize=(15, 12))
                
                decomposition.observed.plot(ax=axes[0], title='Original Time Series')
                decomposition.trend.plot(ax=axes[1], title='Trend')
                decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
                decomposition.resid.plot(ax=axes[3], title='Residual')
                
                plt.tight_layout()
                
                if save_path:
                    decomp_path = save_path.replace('.png', '_decomposition.png')
                    plt.savefig(decomp_path, dpi=300, bbox_inches='tight')
                plt.show()
                
                analysis['seasonality_detected'] = True
                analysis['seasonal_strength'] = np.std(decomposition.seasonal) / np.std(ts)
                
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed: {e}")
                analysis['seasonality_detected'] = False
        
        logger.info(f"Time series analysis completed: {analysis['length']} periods")
        return analysis
    
    def train_arima_model(self, ts: pd.Series = None, order: tuple = (1, 1, 1)) -> ARIMA:
        """
        Train ARIMA model for time series forecasting.
        
        Args:
            ts (pd.Series): Time series data
            order (tuple): ARIMA order (p, d, q)
            
        Returns:
            ARIMA: Trained ARIMA model
        """
        if ts is None:
            ts = self.time_series
            
        if ts is None or len(ts) < 10:
            logger.error("Insufficient data for ARIMA model")
            return None
        
        try:
            # Fit ARIMA model
            model = ARIMA(ts, order=order)
            fitted_model = model.fit()
            
            self.models['arima'] = fitted_model
            
            # Model diagnostics
            logger.info(f"ARIMA{order} model trained successfully")
            logger.info(f"AIC: {fitted_model.aic:.2f}")
            logger.info(f"BIC: {fitted_model.bic:.2f}")
            
            return fitted_model
            
        except Exception as e:
            logger.error(f"ARIMA model training failed: {e}")
            return None
    
    def auto_arima_selection(self, ts: pd.Series = None, max_p: int = 3, 
                           max_d: int = 2, max_q: int = 3) -> tuple:
        """
        Automatically select best ARIMA parameters.
        
        Args:
            ts (pd.Series): Time series data
            max_p (int): Maximum AR order
            max_d (int): Maximum differencing order
            max_q (int): Maximum MA order
            
        Returns:
            tuple: Best ARIMA order
        """
        if ts is None:
            ts = self.time_series
            
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        logger.info("Searching for optimal ARIMA parameters...")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            
                    except Exception:
                        continue
        
        logger.info(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def prepare_lstm_data(self, ts: pd.Series = None, lookback: int = 12, 
                         test_size: float = 0.2) -> tuple:
        """
        Prepare data for LSTM training.
        
        Args:
            ts (pd.Series): Time series data
            lookback (int): Number of previous time steps to use
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        if ts is None:
            ts = self.time_series
            
        if ts is None or len(ts) < lookback + 1:
            logger.error("Insufficient data for LSTM model")
            return None, None, None, None, None
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(ts.values.reshape(-1, 1))
        
        self.scalers['lstm'] = scaler
        
        # Create sequences
        def create_sequences(data, lookback):
            X, y = [], []
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(scaled_data, lookback)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        logger.info(f"LSTM data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None,
                        **kwargs) -> tf.keras.Model:
        """
        Train LSTM model for time series forecasting.
        
        Args:
            X_train (np.ndarray): Training sequences
            y_train (np.ndarray): Training targets
            X_test (np.ndarray): Test sequences
            y_test (np.ndarray): Test targets
            **kwargs: Additional parameters for LSTM model
            
        Returns:
            tf.keras.Model: Trained LSTM model
        """
        # Default parameters
        default_params = {
            'units': 50,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout': 0.2
        }
        
        # Override with config if available
        if self.config.get('models', {}).get('time_series', {}).get('lstm'):
            default_params.update(self.config['models']['time_series']['lstm'])
        
        # Override with provided kwargs
        default_params.update(kwargs)
        
        # Build LSTM model
        model = Sequential([
            LSTM(default_params['units'], return_sequences=True, 
                input_shape=(X_train.shape[1], 1)),
            Dropout(default_params['dropout']),
            LSTM(default_params['units'], return_sequences=False),
            Dropout(default_params['dropout']),
            Dense(25),
            Dense(1)
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=default_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        # Prepare validation data
        validation_data = (X_test, y_test) if X_test is not None and y_test is not None else None
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=default_params['epochs'],
            batch_size=default_params['batch_size'],
            validation_data=validation_data,
            verbose=0
        )
        
        self.models['lstm'] = model
        
        logger.info(f"LSTM model trained successfully ({default_params['epochs']} epochs)")
        
        return model
    
    def forecast_arima(self, model, steps: int = 12) -> tuple:
        """
        Generate forecasts using ARIMA model.
        
        Args:
            model: Trained ARIMA model
            steps (int): Number of steps to forecast
            
        Returns:
            tuple: (forecast, confidence_intervals)
        """
        try:
            forecast_result = model.forecast(steps=steps)
            forecast = forecast_result
            
            # Get confidence intervals
            conf_int = model.get_forecast(steps=steps).conf_int()
            
            # Create forecast index
            last_date = self.time_series.index[-1]
            freq = self.time_series.index.freq or 'M'
            forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                         periods=steps, freq=freq)
            
            forecast_series = pd.Series(forecast, index=forecast_index)
            
            self.forecasts['arima'] = {
                'forecast': forecast_series,
                'conf_int': conf_int
            }
            
            logger.info(f"ARIMA forecast generated for {steps} periods")
            
            return forecast_series, conf_int
            
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}")
            return None, None
    
    def forecast_lstm(self, model, scaler: MinMaxScaler, last_sequence: np.ndarray,
                     steps: int = 12) -> pd.Series:
        """
        Generate forecasts using LSTM model.
        
        Args:
            model: Trained LSTM model
            scaler: Fitted MinMaxScaler
            last_sequence (np.ndarray): Last sequence for prediction
            steps (int): Number of steps to forecast
            
        Returns:
            pd.Series: Forecast values
        """
        try:
            forecasts = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # Predict next value
                next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
                forecasts.append(next_pred[0, 0])
                
                # Update sequence (rolling window)
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred[0, 0]
            
            # Inverse transform forecasts
            forecasts = np.array(forecasts).reshape(-1, 1)
            forecasts = scaler.inverse_transform(forecasts).flatten()
            
            # Create forecast index
            last_date = self.time_series.index[-1]
            freq = self.time_series.index.freq or 'M'
            forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                         periods=steps, freq=freq)
            
            forecast_series = pd.Series(forecasts, index=forecast_index)
            
            self.forecasts['lstm'] = forecast_series
            
            logger.info(f"LSTM forecast generated for {steps} periods")
            
            return forecast_series
            
        except Exception as e:
            logger.error(f"LSTM forecasting failed: {e}")
            return None
    
    def evaluate_forecasts(self, actual: pd.Series, forecast: pd.Series, 
                          model_name: str = "Model") -> dict:
        """
        Evaluate forecast accuracy.
        
        Args:
            actual (pd.Series): Actual values
            forecast (pd.Series): Forecasted values
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        # Align series
        common_index = actual.index.intersection(forecast.index)
        if len(common_index) == 0:
            logger.warning("No common dates for evaluation")
            return {}
        
        actual_aligned = actual[common_index]
        forecast_aligned = forecast[common_index]
        
        # Calculate metrics
        mse = mean_squared_error(actual_aligned, forecast_aligned)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_aligned, forecast_aligned)
        mape = np.mean(np.abs((actual_aligned - forecast_aligned) / actual_aligned)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'n_periods': len(common_index)
        }
        
        logger.info(f"{model_name} Forecast Evaluation:")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def plot_forecasts(self, save_path: str = None) -> None:
        """
        Plot time series and forecasts.
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.forecasts:
            logger.warning("No forecasts available to plot")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot historical data
        plt.plot(self.time_series.index, self.time_series.values, 
                label='Historical Data', color='blue', linewidth=2)
        
        # Plot forecasts
        colors = ['red', 'green', 'purple', 'orange']
        for i, (model_name, forecast_data) in enumerate(self.forecasts.items()):
            color = colors[i % len(colors)]
            
            if isinstance(forecast_data, dict):
                # ARIMA with confidence intervals
                forecast = forecast_data['forecast']
                plt.plot(forecast.index, forecast.values, 
                        label=f'{model_name.upper()} Forecast', 
                        color=color, linewidth=2, linestyle='--')
                
                if 'conf_int' in forecast_data:
                    conf_int = forecast_data['conf_int']
                    plt.fill_between(forecast.index, 
                                   conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                                   alpha=0.2, color=color, 
                                   label=f'{model_name.upper()} Confidence Interval')
            else:
                # Simple forecast
                plt.plot(forecast_data.index, forecast_data.values, 
                        label=f'{model_name.upper()} Forecast', 
                        color=color, linewidth=2, linestyle='--')
        
        plt.title('Crime Count Time Series and Forecasts', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Crime Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_forecasting_models(self, output_dir: str = "models/saved_models") -> None:
        """
        Save forecasting models and scalers.
        
        Args:
            output_dir (str): Directory to save models
        """
        ensure_dir_exists(output_dir)
        
        for model_name, model in self.models.items():
            if model_name == 'arima':
                model_path = f"{output_dir}/forecasting_{model_name}_model.pkl"
                save_model(model, model_path)
            elif model_name == 'lstm':
                model_path = f"{output_dir}/forecasting_{model_name}_model.h5"
                model.save(model_path)
        
        for scaler_name, scaler in self.scalers.items():
            scaler_path = f"{output_dir}/forecasting_{scaler_name}_scaler.pkl"
            save_model(scaler, scaler_path)
        
        logger.info(f"Forecasting models saved to {output_dir}")
    
    def generate_forecasting_report(self, df: pd.DataFrame = None, 
                                   output_dir: str = "reports/forecasting") -> dict:
        """
        Generate comprehensive forecasting report.
        
        Args:
            df (pd.DataFrame): Input dataframe
            output_dir (str): Directory to save reports
            
        Returns:
            dict: Comprehensive forecasting analysis
        """
        if df is None:
            df = self.df
            
        ensure_dir_exists(output_dir)
        
        # Prepare time series
        ts = self.prepare_time_series(df, freq="M")
        
        # Analyze time series
        analysis = self.analyze_time_series(ts, f"{output_dir}/time_series_analysis.png")
        
        # Split data for evaluation
        split_idx = int(len(ts) * 0.8)
        train_ts = ts[:split_idx]
        test_ts = ts[split_idx:]
        
        # Train ARIMA model
        best_order = self.auto_arima_selection(train_ts)
        arima_model = self.train_arima_model(train_ts, best_order)
        
        # Train LSTM model
        X_train, X_test, y_train, y_test, scaler = self.prepare_lstm_data(train_ts)
        if X_train is not None:
            lstm_model = self.train_lstm_model(X_train, y_train, X_test, y_test)
        
        # Generate forecasts
        forecast_steps = len(test_ts)
        
        if arima_model:
            arima_forecast, arima_conf = self.forecast_arima(arima_model, forecast_steps)
        
        if 'lstm' in self.models and X_train is not None:
            last_sequence = X_train[-1].flatten()
            lstm_forecast = self.forecast_lstm(self.models['lstm'], scaler, 
                                             last_sequence, forecast_steps)
        
        # Evaluate forecasts
        evaluation = {}
        if 'arima' in self.forecasts:
            evaluation['arima'] = self.evaluate_forecasts(test_ts, 
                                                         self.forecasts['arima']['forecast'], 
                                                         "ARIMA")
        
        if 'lstm' in self.forecasts:
            evaluation['lstm'] = self.evaluate_forecasts(test_ts, 
                                                        self.forecasts['lstm'], 
                                                        "LSTM")
        
        # Plot forecasts
        self.plot_forecasts(f"{output_dir}/forecasts_comparison.png")
        
        # Generate future forecasts
        future_steps = 12
        if arima_model:
            future_arima, _ = self.forecast_arima(arima_model, future_steps)
        
        if 'lstm' in self.models and X_train is not None:
            # Use full data for future prediction
            full_X, _, _, _, full_scaler = self.prepare_lstm_data(ts, test_size=0.0)
            if full_X is not None:
                last_sequence = full_X[-1].flatten()
                future_lstm = self.forecast_lstm(self.models['lstm'], full_scaler, 
                                               last_sequence, future_steps)
        
        # Save models
        self.save_forecasting_models()
        
        # Compile report
        report = {
            'time_series_analysis': analysis,
            'model_evaluation': evaluation,
            'future_forecasts': {
                'arima': future_arima.to_dict() if 'arima' in self.forecasts else None,
                'lstm': future_lstm.to_dict() if 'lstm' in self.forecasts else None
            },
            'recommendations': self._generate_forecasting_recommendations(analysis, evaluation)
        }
        
        logger.info(f"Comprehensive forecasting report generated in {output_dir}")
        return report
    
    def _generate_forecasting_recommendations(self, analysis: dict, evaluation: dict) -> list:
        """
        Generate recommendations based on forecasting analysis.
        
        Args:
            analysis (dict): Time series analysis results
            evaluation (dict): Model evaluation results
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # Data quality recommendations
        if analysis.get('length', 0) < 24:
            recommendations.append(
                "Time series is too short for reliable forecasting. "
                "Collect more historical data for better predictions."
            )
        
        # Seasonality recommendations
        if analysis.get('seasonality_detected', False):
            recommendations.append(
                "Seasonal patterns detected. Consider seasonal ARIMA (SARIMA) "
                "or seasonal adjustments for better forecasting."
            )
        
        # Model performance recommendations
        if evaluation:
            valid_models = {k: v for k, v in evaluation.items() if v and 'rmse' in v}
            
            if valid_models:
                best_model = min(valid_models.keys(), 
                               key=lambda x: valid_models[x].get('rmse', float('inf')))
                best_rmse = valid_models[best_model]['rmse']
                
                recommendations.append(
                    f"{best_model.upper()} model performs best with RMSE: {best_rmse:.2f}. "
                    "Consider using this model for production forecasting."
                )
                
                # Check for high error rates
                for model_name, metrics in valid_models.items():
                    mape = metrics.get('mape', 0)
                    if mape > 20:
                        recommendations.append(
                            f"{model_name.upper()} model has high error rate (MAPE: {mape:.1f}%). "
                            "Consider model improvement or additional features."
                        )
            else:
                recommendations.append(
                    "Model evaluation data is insufficient. "
                    "Ensure proper train-test split for reliable evaluation."
                )
        
        # General recommendations
        recommendations.extend([
            "Regularly update forecasting models with new data to maintain accuracy.",
            "Monitor forecast accuracy and adjust models when performance degrades.",
            "Consider ensemble forecasting combining multiple models for robustness.",
            "Implement automated retraining pipelines for production systems."
        ])
        
        return recommendations

def main():
    """
    Main function to demonstrate forecasting capabilities.
    """
    # Initialize forecaster
    forecaster = CrimeTimeSeriesForecaster()
    
    # Load processed data
    data_path = "data/processed/crime_data_processed.csv"
    if os.path.exists(data_path):
        df = forecaster.load_data(data_path)
    else:
        # Create sample data if processed data doesn't exist
        from src.data_preprocessing import CrimeDataPreprocessor
        from src.utils import create_sample_crime_data, create_sample_demographic_data
        
        crime_data = create_sample_crime_data(1000)
        demographic_data = create_sample_demographic_data()
        
        preprocessor = CrimeDataPreprocessor()
        df = preprocessor.preprocess_pipeline(crime_data, demographic_data)
    
    # Generate comprehensive forecasting report
    report = forecaster.generate_forecasting_report(df)
    
    print("\n" + "="*50)
    print("CRIME FORECASTING REPORT")
    print("="*50)
    
    print(f"\nTime Series Analysis:")
    analysis = report.get('time_series_analysis', {})
    print(f"  Data Length: {analysis.get('length', 'N/A')} periods")
    print(f"  Date Range: {analysis.get('start_date', 'N/A')} to {analysis.get('end_date', 'N/A')}")
    print(f"  Mean Crime Count: {analysis.get('mean', 0):.2f}")
    print(f"  Trend: {analysis.get('trend', 'Unknown')}")
    print(f"  Seasonality: {'Detected' if analysis.get('seasonality_detected', False) else 'Not detected'}")
    
    print(f"\nModel Evaluation:")
    evaluation = report.get('model_evaluation', {})
    for model_name, metrics in evaluation.items():
        print(f"  {model_name.upper()}:")
        print(f"    RMSE: {metrics.get('rmse', 0):.2f}")
        print(f"    MAE: {metrics.get('mae', 0):.2f}")
        print(f"    MAPE: {metrics.get('mape', 0):.2f}%")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report.get('recommendations', []), 1):
        print(f"  {i}. {rec}")
    
    print("\nForecasting visualizations saved to 'reports/forecasting/' directory")

if __name__ == "__main__":
    main()