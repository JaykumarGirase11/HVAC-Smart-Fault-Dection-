"""
Feature Engineering for HVAC Fault Detection
Extracts and creates relevant features from raw sensor data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class HVACFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def create_derived_features(self, df):
        """Create engineered features from raw sensor data"""
        df = df.copy()
        
        # Temperature differentials
        df['temp_differential'] = df['return_air_temp'] - df['supply_air_temp']
        df['outdoor_supply_diff'] = df['outdoor_temp'] - df['supply_air_temp']
        
        # Efficiency metrics
        df['cooling_efficiency'] = df['temp_differential'] / (df['power_consumption'] + 0.1)
        df['power_per_airflow'] = df['power_consumption'] / (df['airflow_rate'] + 0.1)
        
        # Pressure-related features
        df['pressure_per_runtime'] = df['refrigerant_pressure'] / (df['compressor_runtime'] + 0.1)
        
        # Airflow efficiency
        df['airflow_per_power'] = df['airflow_rate'] / (df['power_consumption'] + 0.1)
        
        # Temperature ratios
        df['supply_return_ratio'] = df['supply_air_temp'] / (df['return_air_temp'] + 0.1)
        
        # Load indicators
        df['cooling_load'] = (df['outdoor_temp'] - 70) * df['airflow_rate'] / 1000
        
        # Runtime efficiency
        df['runtime_efficiency'] = df['compressor_runtime'] / (df['power_consumption'] + 0.1)
        
        # Pressure anomaly indicators
        df['pressure_deviation'] = np.abs(df['refrigerant_pressure'] - 130)
        
        return df
    
    def create_rolling_features(self, df, window=5):
        """Create rolling window features (for time-series data)"""
        df = df.copy()
        
        if len(df) < window:
            return df
        
        # Rolling means
        df['supply_temp_rolling_mean'] = df['supply_air_temp'].rolling(window=window, min_periods=1).mean()
        df['power_rolling_mean'] = df['power_consumption'].rolling(window=window, min_periods=1).mean()
        df['pressure_rolling_mean'] = df['refrigerant_pressure'].rolling(window=window, min_periods=1).mean()
        
        # Rolling standard deviations (detect instability)
        df['supply_temp_rolling_std'] = df['supply_air_temp'].rolling(window=window, min_periods=1).std().fillna(0)
        df['pressure_rolling_std'] = df['refrigerant_pressure'].rolling(window=window, min_periods=1).std().fillna(0)
        
        return df
    
    def prepare_features(self, df, fit_scaler=False):
        """Prepare features for model training/prediction"""
        # Create derived features
        df = self.create_derived_features(df)
        
        # Select feature columns (exclude target and metadata)
        feature_cols = [col for col in df.columns 
                       if col not in ['fault_type', 'timestamp']]
        
        X = df[feature_cols].copy()
        
        # Handle any infinite or NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = feature_cols
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
    def get_feature_importance_names(self):
        """Return feature names for importance analysis"""
        return self.feature_names

if __name__ == "__main__":
    # Test feature engineering
    df = pd.read_csv('hvac_data.csv')
    
    engineer = HVACFeatureEngineer()
    X = engineer.prepare_features(df, fit_scaler=True)
    
    print("Original features:", df.shape[1])
    print("Engineered features:", X.shape[1])
    print("\nFeature names:")
    print(X.columns.tolist())
    print("\nSample features:")
    print(X.head())
