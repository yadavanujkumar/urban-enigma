"""
Hotspot detection module for crime data using clustering algorithms.
Implements K-Means and DBSCAN for identifying crime concentration areas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, ensure_dir_exists, save_model, load_model
import logging

logger = logging.getLogger(__name__)

class CrimeHotspotDetector:
    """
    Crime hotspot detection using clustering algorithms.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize hotspot detector with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = load_config(config_path) if os.path.exists(config_path) else {}
        self.df = None
        self.clustered_data = None
        self.models = {}
        self.scalers = {}
        
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
    
    def prepare_clustering_features(self, df: pd.DataFrame = None, feature_set: str = "location") -> np.ndarray:
        """
        Prepare features for clustering analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_set (str): Type of features to use ('location', 'comprehensive')
            
        Returns:
            np.ndarray: Prepared feature matrix
        """
        if df is None:
            df = self.df
            
        if feature_set == "location":
            # Use only location-based features
            features = ['latitude', 'longitude']
        elif feature_set == "comprehensive":
            # Use location + temporal + demographic features
            features = ['latitude', 'longitude', 'month', 'weekday', 'weekend', 
                       'population_normalized', 'literacy_rate_normalized', 
                       'unemployment_rate_normalized', 'urban_population_pct_normalized']
        else:
            # Custom feature set
            features = ['latitude', 'longitude', 'state_crime_count_normalized']
        
        # Select available features
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            logger.warning("No suitable features found for clustering")
            return np.array([])
        
        feature_matrix = df[available_features].fillna(0).values
        
        # Scale the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        self.scalers[feature_set] = scaler
        logger.info(f"Prepared clustering features: {available_features}")
        
        return scaled_features
    
    def kmeans_clustering(self, features: np.ndarray, n_clusters: int = 8, 
                         random_state: int = 42) -> np.ndarray:
        """
        Perform K-Means clustering.
        
        Args:
            features (np.ndarray): Feature matrix
            n_clusters (int): Number of clusters
            random_state (int): Random state for reproducibility
            
        Returns:
            np.ndarray: Cluster labels
        """
        if len(features) == 0:
            return np.array([])
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        self.models['kmeans'] = kmeans
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(features, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(features, cluster_labels)
        
        logger.info(f"K-Means clustering completed with {n_clusters} clusters")
        logger.info(f"Silhouette Score: {silhouette_avg:.3f}")
        logger.info(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
        
        return cluster_labels
    
    def dbscan_clustering(self, features: np.ndarray, eps: float = 0.5, 
                         min_samples: int = 5) -> np.ndarray:
        """
        Perform DBSCAN clustering.
        
        Args:
            features (np.ndarray): Feature matrix
            eps (float): Maximum distance between samples
            min_samples (int): Minimum samples in a neighborhood
            
        Returns:
            np.ndarray: Cluster labels
        """
        if len(features) == 0:
            return np.array([])
            
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features)
        
        self.models['dbscan'] = dbscan
        
        # Calculate metrics (excluding noise points for silhouette score)
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) > 1 and len(np.unique(cluster_labels[non_noise_mask])) > 1:
            silhouette_avg = silhouette_score(features[non_noise_mask], 
                                            cluster_labels[non_noise_mask])
            calinski_harabasz = calinski_harabasz_score(features[non_noise_mask], 
                                                       cluster_labels[non_noise_mask])
        else:
            silhouette_avg = -1
            calinski_harabasz = -1
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"DBSCAN clustering completed with {n_clusters} clusters and {n_noise} noise points")
        logger.info(f"Silhouette Score: {silhouette_avg:.3f}")
        logger.info(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
        
        return cluster_labels
    
    def optimize_kmeans_clusters(self, features: np.ndarray, max_clusters: int = 15) -> dict:
        """
        Find optimal number of clusters for K-Means using elbow method and silhouette score.
        
        Args:
            features (np.ndarray): Feature matrix
            max_clusters (int): Maximum number of clusters to test
            
        Returns:
            dict: Optimization results
        """
        if len(features) == 0:
            return {}
            
        inertias = []
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find elbow point (simplified)
        best_silhouette_k = cluster_range[np.argmax(silhouette_scores)]
        
        results = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'best_k_silhouette': best_silhouette_k,
            'best_silhouette_score': max(silhouette_scores)
        }
        
        logger.info(f"Optimal K-Means clusters: {best_silhouette_k} (Silhouette: {max(silhouette_scores):.3f})")
        
        return results
    
    def detect_hotspots(self, df: pd.DataFrame = None, method: str = "kmeans", 
                       feature_set: str = "location", **kwargs) -> pd.DataFrame:
        """
        Detect crime hotspots using specified clustering method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Clustering method ('kmeans' or 'dbscan')
            feature_set (str): Feature set to use for clustering
            **kwargs: Additional arguments for clustering methods
            
        Returns:
            pd.DataFrame: Dataframe with cluster labels
        """
        if df is None:
            df = self.df
            
        # Prepare features
        features = self.prepare_clustering_features(df, feature_set)
        
        if len(features) == 0:
            logger.error("No features available for clustering")
            return df
        
        # Perform clustering
        if method.lower() == "kmeans":
            n_clusters = kwargs.get('n_clusters', 8)
            cluster_labels = self.kmeans_clustering(features, n_clusters)
        elif method.lower() == "dbscan":
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            cluster_labels = self.dbscan_clustering(features, eps, min_samples)
        else:
            logger.error(f"Unknown clustering method: {method}")
            return df
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered[f'{method}_cluster'] = cluster_labels
        df_clustered['is_hotspot'] = cluster_labels != -1  # For DBSCAN, -1 indicates noise
        
        self.clustered_data = df_clustered
        
        logger.info(f"Hotspot detection completed using {method}")
        return df_clustered
    
    def plot_clustering_optimization(self, optimization_results: dict, save_path: str = None) -> None:
        """
        Plot clustering optimization results.
        
        Args:
            optimization_results (dict): Results from optimize_kmeans_clusters
            save_path (str): Path to save the plot
        """
        if not optimization_results:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow curve
        ax1.plot(optimization_results['cluster_range'], 
                optimization_results['inertias'], 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(optimization_results['cluster_range'], 
                optimization_results['silhouette_scores'], 'ro-')
        ax2.axvline(x=optimization_results['best_k_silhouette'], 
                   color='green', linestyle='--', 
                   label=f"Best k={optimization_results['best_k_silhouette']}")
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hotspots_2d(self, df: pd.DataFrame = None, method: str = "kmeans", 
                        save_path: str = None) -> None:
        """
        Plot 2D visualization of detected hotspots.
        
        Args:
            df (pd.DataFrame): Clustered dataframe
            method (str): Clustering method used
            save_path (str): Path to save the plot
        """
        if df is None:
            df = self.clustered_data
            
        if df is None or f'{method}_cluster' not in df.columns:
            logger.error("No clustering results found")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Geographic scatter plot
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(df['longitude'], df['latitude'], 
                            c=df[f'{method}_cluster'], 
                            cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Crime Hotspots - {method.upper()} Clustering')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Cluster size distribution
        plt.subplot(2, 2, 2)
        cluster_counts = df[f'{method}_cluster'].value_counts().sort_index()
        cluster_counts.plot(kind='bar', color='skyblue')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Crimes')
        plt.title('Crimes per Cluster')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Crime types by cluster
        plt.subplot(2, 2, 3)
        if 'crime_type' in df.columns:
            cluster_crime_pivot = df.groupby([f'{method}_cluster', 'crime_type']).size().unstack(fill_value=0)
            cluster_crime_pivot.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Crimes')
            plt.title('Crime Types by Cluster')
            plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0)
        
        # Plot 4: Time distribution by cluster
        plt.subplot(2, 2, 4)
        if 'month' in df.columns:
            cluster_month_pivot = df.groupby([f'{method}_cluster', 'month']).size().unstack(fill_value=0)
            cluster_month_pivot.T.plot(kind='line', marker='o', ax=plt.gca())
            plt.xlabel('Month')
            plt.ylabel('Number of Crimes')
            plt.title('Monthly Crime Distribution by Cluster')
            plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_folium_heatmap(self, df: pd.DataFrame = None, save_path: str = None) -> folium.Map:
        """
        Create interactive heatmap using Folium.
        
        Args:
            df (pd.DataFrame): Crime dataframe
            save_path (str): Path to save the HTML file
            
        Returns:
            folium.Map: Folium map object
        """
        if df is None:
            df = self.df
            
        # Center the map on India
        india_center = [20.5937, 78.9629]
        
        # Create base map
        m = folium.Map(location=india_center, zoom_start=5, tiles='OpenStreetMap')
        
        # Prepare heat data
        heat_data = []
        for idx, row in df.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                heat_data.append([row['latitude'], row['longitude']])
        
        # Add heatmap layer
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
        
        # Add cluster markers if available
        if hasattr(self, 'clustered_data') and self.clustered_data is not None:
            cluster_column = None
            for col in self.clustered_data.columns:
                if 'cluster' in col and col != 'district_crime_count':
                    cluster_column = col
                    break
            
            if cluster_column:
                # Add cluster centers
                cluster_centers = self.clustered_data.groupby(cluster_column)[['latitude', 'longitude']].mean()
                
                for cluster_id, center in cluster_centers.iterrows():
                    if cluster_id != -1:  # Skip noise points for DBSCAN
                        cluster_size = len(self.clustered_data[self.clustered_data[cluster_column] == cluster_id])
                        
                        folium.CircleMarker(
                            location=[center['latitude'], center['longitude']],
                            radius=max(5, min(20, cluster_size / 10)),
                            popup=f'Cluster {cluster_id}<br>Crimes: {cluster_size}',
                            color='red',
                            fill=True,
                            fillColor='red',
                            fillOpacity=0.6
                        ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <b>Crime Heatmap</b><br>
        <i class="fa fa-circle" style="color:red"></i> Cluster Centers<br>
        <i class="fa fa-square" style="color:blue"></i> Heat Intensity
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(save_path)
            logger.info(f"Interactive heatmap saved to {save_path}")
        
        return m
    
    def create_plotly_3d_clusters(self, df: pd.DataFrame = None, method: str = "kmeans", 
                                 save_path: str = None) -> None:
        """
        Create 3D visualization of clusters using Plotly.
        
        Args:
            df (pd.DataFrame): Clustered dataframe
            method (str): Clustering method used
            save_path (str): Path to save the HTML file
        """
        if df is None:
            df = self.clustered_data
            
        if df is None or f'{method}_cluster' not in df.columns:
            logger.error("No clustering results found")
            return
        
        # Sample data for better visualization
        if len(df) > 1000:
            df_sample = df.sample(1000)
        else:
            df_sample = df
        
        fig = go.Figure(data=[go.Scatter3d(
            x=df_sample['longitude'],
            y=df_sample['latitude'],
            z=df_sample['month'],
            mode='markers',
            marker=dict(
                size=5,
                color=df_sample[f'{method}_cluster'],
                colorscale='viridis',
                opacity=0.8,
                colorbar=dict(title="Cluster")
            ),
            text=df_sample['state'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Longitude: %{x}<br>' +
                         'Latitude: %{y}<br>' +
                         'Month: %{z}<br>' +
                         'Cluster: %{marker.color}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'3D Crime Hotspots - {method.upper()} Clustering',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Month'
            ),
            width=800,
            height=600
        )
        
        fig.show()
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"3D cluster visualization saved to {save_path}")
    
    def analyze_hotspot_characteristics(self, df: pd.DataFrame = None, 
                                      method: str = "kmeans") -> dict:
        """
        Analyze characteristics of detected hotspots.
        
        Args:
            df (pd.DataFrame): Clustered dataframe
            method (str): Clustering method used
            
        Returns:
            dict: Hotspot analysis results
        """
        if df is None:
            df = self.clustered_data
            
        if df is None or f'{method}_cluster' not in df.columns:
            logger.error("No clustering results found")
            return {}
        
        cluster_column = f'{method}_cluster'
        analysis = {}
        
        # Overall statistics
        total_clusters = df[cluster_column].nunique()
        if method == 'dbscan':
            total_clusters -= 1 if -1 in df[cluster_column].values else 0
        
        analysis['total_clusters'] = total_clusters
        analysis['total_crimes'] = len(df)
        analysis['clustered_crimes'] = len(df[df[cluster_column] != -1])
        
        # Cluster-wise analysis
        cluster_stats = []
        for cluster_id in sorted(df[cluster_column].unique()):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = df[df[cluster_column] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'crime_count': len(cluster_data),
                'crime_percentage': len(cluster_data) / len(df) * 100,
                'center_lat': cluster_data['latitude'].mean(),
                'center_lon': cluster_data['longitude'].mean(),
                'most_common_crime': cluster_data['crime_type'].mode().iloc[0] if 'crime_type' in cluster_data.columns else 'Unknown',
                'most_affected_state': cluster_data['state'].mode().iloc[0] if 'state' in cluster_data.columns else 'Unknown',
                'peak_month': cluster_data['month'].mode().iloc[0] if 'month' in cluster_data.columns else 'Unknown'
            }
            
            cluster_stats.append(stats)
        
        analysis['cluster_details'] = cluster_stats
        
        # Find largest hotspots
        if cluster_stats:
            largest_cluster = max(cluster_stats, key=lambda x: x['crime_count'])
            analysis['largest_hotspot'] = largest_cluster
        
        logger.info(f"Hotspot analysis completed for {total_clusters} clusters")
        return analysis
    
    def save_clustering_models(self, output_dir: str = "models/saved_models") -> None:
        """
        Save clustering models and scalers.
        
        Args:
            output_dir (str): Directory to save models
        """
        ensure_dir_exists(output_dir)
        
        for model_name, model in self.models.items():
            model_path = f"{output_dir}/hotspot_{model_name}_model.pkl"
            save_model(model, model_path)
        
        for scaler_name, scaler in self.scalers.items():
            scaler_path = f"{output_dir}/hotspot_{scaler_name}_scaler.pkl"
            save_model(scaler, scaler_path)
        
        logger.info(f"Clustering models saved to {output_dir}")
    
    def generate_hotspot_report(self, df: pd.DataFrame = None, 
                               output_dir: str = "reports/hotspots") -> dict:
        """
        Generate comprehensive hotspot analysis report.
        
        Args:
            df (pd.DataFrame): Input dataframe
            output_dir (str): Directory to save reports
            
        Returns:
            dict: Comprehensive hotspot analysis
        """
        if df is None:
            df = self.df
            
        ensure_dir_exists(output_dir)
        
        # Optimize K-Means clusters
        features = self.prepare_clustering_features(df, "location")
        optimization_results = self.optimize_kmeans_clusters(features)
        
        # Perform clustering with optimal parameters
        optimal_k = optimization_results.get('best_k_silhouette', 8)
        df_kmeans = self.detect_hotspots(df, method="kmeans", 
                                        feature_set="location", 
                                        n_clusters=optimal_k)
        
        df_dbscan = self.detect_hotspots(df, method="dbscan", 
                                        feature_set="location", 
                                        eps=0.3, min_samples=10)
        
        # Generate visualizations
        self.plot_clustering_optimization(optimization_results, 
                                        f"{output_dir}/clustering_optimization.png")
        
        self.plot_hotspots_2d(df_kmeans, "kmeans", 
                             f"{output_dir}/kmeans_hotspots.png")
        
        self.plot_hotspots_2d(df_dbscan, "dbscan", 
                             f"{output_dir}/dbscan_hotspots.png")
        
        self.create_folium_heatmap(df, f"{output_dir}/interactive_heatmap.html")
        self.create_plotly_3d_clusters(df_kmeans, "kmeans", 
                                      f"{output_dir}/3d_clusters.html")
        
        # Analyze hotspots
        kmeans_analysis = self.analyze_hotspot_characteristics(df_kmeans, "kmeans")
        dbscan_analysis = self.analyze_hotspot_characteristics(df_dbscan, "dbscan")
        
        # Save models
        self.save_clustering_models()
        
        # Compile report
        report = {
            'optimization_results': optimization_results,
            'kmeans_analysis': kmeans_analysis,
            'dbscan_analysis': dbscan_analysis,
            'recommendations': self._generate_recommendations(kmeans_analysis, dbscan_analysis)
        }
        
        logger.info(f"Comprehensive hotspot report generated in {output_dir}")
        return report
    
    def _generate_recommendations(self, kmeans_analysis: dict, dbscan_analysis: dict) -> list:
        """
        Generate recommendations based on hotspot analysis.
        
        Args:
            kmeans_analysis (dict): K-Means analysis results
            dbscan_analysis (dict): DBSCAN analysis results
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # K-Means recommendations
        if kmeans_analysis.get('largest_hotspot'):
            largest = kmeans_analysis['largest_hotspot']
            recommendations.append(
                f"Priority hotspot: {largest['most_affected_state']} with {largest['crime_count']} crimes "
                f"({largest['crime_percentage']:.1f}% of total). Focus on {largest['most_common_crime']} prevention."
            )
        
        # DBSCAN recommendations
        if dbscan_analysis.get('total_clusters', 0) > 0:
            recommendations.append(
                f"DBSCAN identified {dbscan_analysis['total_clusters']} distinct crime clusters. "
                f"{dbscan_analysis['clustered_crimes']} crimes ({dbscan_analysis['clustered_crimes']/dbscan_analysis['total_crimes']*100:.1f}%) "
                "are part of concentrated hotspots."
            )
        
        # General recommendations
        recommendations.extend([
            "Deploy additional police patrols in identified hotspot areas during peak crime months.",
            "Implement targeted crime prevention programs in high-density clusters.",
            "Consider socio-economic interventions in areas with persistent hotspots.",
            "Use real-time monitoring systems in the top 3 highest-crime clusters."
        ])
        
        return recommendations

def main():
    """
    Main function to demonstrate hotspot detection capabilities.
    """
    # Initialize detector
    detector = CrimeHotspotDetector()
    
    # Load processed data
    data_path = "data/processed/crime_data_processed.csv"
    if os.path.exists(data_path):
        df = detector.load_data(data_path)
    else:
        # Create sample data if processed data doesn't exist
        from src.data_preprocessing import CrimeDataPreprocessor
        from src.utils import create_sample_crime_data, create_sample_demographic_data
        
        crime_data = create_sample_crime_data(1000)
        demographic_data = create_sample_demographic_data()
        
        preprocessor = CrimeDataPreprocessor()
        df = preprocessor.preprocess_pipeline(crime_data, demographic_data)
    
    # Generate comprehensive hotspot report
    report = detector.generate_hotspot_report(df)
    
    print("\n" + "="*50)
    print("CRIME HOTSPOT DETECTION REPORT")
    print("="*50)
    
    print(f"\nOptimization Results:")
    if report.get('optimization_results'):
        opt = report['optimization_results']
        print(f"  Optimal K-Means clusters: {opt.get('best_k_silhouette', 'N/A')}")
        print(f"  Best silhouette score: {opt.get('best_silhouette_score', 'N/A'):.3f}")
    
    print(f"\nK-Means Analysis:")
    if report.get('kmeans_analysis'):
        kmeans = report['kmeans_analysis']
        print(f"  Total clusters: {kmeans.get('total_clusters', 'N/A')}")
        print(f"  Clustered crimes: {kmeans.get('clustered_crimes', 'N/A')}")
        if kmeans.get('largest_hotspot'):
            largest = kmeans['largest_hotspot']
            print(f"  Largest hotspot: {largest['most_affected_state']} ({largest['crime_count']} crimes)")
    
    print(f"\nDBSCAN Analysis:")
    if report.get('dbscan_analysis'):
        dbscan = report['dbscan_analysis']
        print(f"  Total clusters: {dbscan.get('total_clusters', 'N/A')}")
        print(f"  Clustered crimes: {dbscan.get('clustered_crimes', 'N/A')}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report.get('recommendations', []), 1):
        print(f"  {i}. {rec}")
    
    print("\nHotspot visualizations saved to 'reports/hotspots/' directory")

if __name__ == "__main__":
    main()