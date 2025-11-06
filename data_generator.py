"""
HVAC Synthetic Data Generator
Generates realistic HVAC operational data with various fault patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class HVACDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        # Normal operating ranges
        self.normal_ranges = {
            'supply_air_temp': (55, 65),  # °F
            'return_air_temp': (70, 78),  # °F
            'outdoor_temp': (60, 95),  # °F
            'refrigerant_pressure': (120, 140),  # PSI
            'power_consumption': (8, 12),  # kW
            'airflow_rate': (1800, 2200),  # CFM
            'compressor_runtime': (40, 60),  # % of time
        }
        
        # Fault patterns
        self.fault_types = [
            'Normal',
            'Refrigerant_Leak',
            'Sensor_Drift',
            'Compressor_Failure',
            'Air_Filter_Clog',
            'Evaporator_Fouling',
            'Condenser_Fouling',
            'Expansion_Valve_Stuck',
            'Fan_Motor_Issue'
        ]
    
    def generate_normal_data(self, n_samples):
        """Generate normal operating data"""
        data = {
            'supply_air_temp': np.random.uniform(*self.normal_ranges['supply_air_temp'], n_samples),
            'return_air_temp': np.random.uniform(*self.normal_ranges['return_air_temp'], n_samples),
            'outdoor_temp': np.random.uniform(*self.normal_ranges['outdoor_temp'], n_samples),
            'refrigerant_pressure': np.random.uniform(*self.normal_ranges['refrigerant_pressure'], n_samples),
            'power_consumption': np.random.uniform(*self.normal_ranges['power_consumption'], n_samples),
            'airflow_rate': np.random.uniform(*self.normal_ranges['airflow_rate'], n_samples),
            'compressor_runtime': np.random.uniform(*self.normal_ranges['compressor_runtime'], n_samples),
        }
        
        # Add small correlations for realism
        data['supply_air_temp'] += (data['outdoor_temp'] - 75) * 0.1
        data['power_consumption'] += (data['outdoor_temp'] - 75) * 0.05
        
        return data
    
    def inject_refrigerant_leak(self, data, severity=0.7):
        """Simulate refrigerant leak - low pressure, poor cooling"""
        data['refrigerant_pressure'] *= (1 - severity * 0.4)
        data['supply_air_temp'] += severity * 8
        data['compressor_runtime'] += severity * 15
        data['power_consumption'] += severity * 2
        return data
    
    def inject_sensor_drift(self, data, severity=0.7):
        """Simulate sensor drift - inconsistent readings"""
        if isinstance(data['supply_air_temp'], (int, float)):
            drift = np.random.normal(0, severity * 5)
            data['supply_air_temp'] += drift
            data['return_air_temp'] += drift * 0.8
        else:
            drift = np.random.normal(0, severity * 5, len(data['supply_air_temp']))
            data['supply_air_temp'] += drift
            data['return_air_temp'] += drift * 0.8
        return data
    
    def inject_compressor_failure(self, data, severity=0.7):
        """Simulate compressor failure - high power, low cooling"""
        data['power_consumption'] += severity * 5
        data['supply_air_temp'] += severity * 10
        data['refrigerant_pressure'] -= severity * 20
        data['compressor_runtime'] += severity * 20
        return data
    
    def inject_air_filter_clog(self, data, severity=0.7):
        """Simulate clogged air filter - reduced airflow"""
        data['airflow_rate'] *= (1 - severity * 0.35)
        data['supply_air_temp'] += severity * 4
        data['power_consumption'] += severity * 1.5
        return data
    
    def inject_evaporator_fouling(self, data, severity=0.7):
        """Simulate evaporator fouling - reduced heat transfer"""
        data['supply_air_temp'] += severity * 6
        data['refrigerant_pressure'] += severity * 15
        data['airflow_rate'] *= (1 - severity * 0.15)
        return data
    
    def inject_condenser_fouling(self, data, severity=0.7):
        """Simulate condenser fouling - high pressure, poor efficiency"""
        data['refrigerant_pressure'] += severity * 25
        data['power_consumption'] += severity * 3
        data['supply_air_temp'] += severity * 5
        return data
    
    def inject_expansion_valve_stuck(self, data, severity=0.7):
        """Simulate stuck expansion valve - pressure issues"""
        data['refrigerant_pressure'] += severity * 30
        data['supply_air_temp'] += severity * 7
        data['compressor_runtime'] += severity * 10
        return data
    
    def inject_fan_motor_issue(self, data, severity=0.7):
        """Simulate fan motor problems - airflow issues"""
        data['airflow_rate'] *= (1 - severity * 0.4)
        data['power_consumption'] *= (1 - severity * 0.2)
        data['supply_air_temp'] += severity * 6
        return data
    
    def generate_dataset(self, hours=25000, samples_per_hour=1):
        """Generate complete dataset with all fault types"""
        total_samples = hours * samples_per_hour
        samples_per_fault = total_samples // len(self.fault_types)
        
        all_data = []
        start_time = datetime.now() - timedelta(hours=hours)
        
        for fault_idx, fault_type in enumerate(self.fault_types):
            print(f"Generating {fault_type} data...")
            
            # Generate base normal data
            data = self.generate_normal_data(samples_per_fault)
            
            # Inject faults with varying severity
            if fault_type != 'Normal':
                severity = np.random.uniform(0.5, 0.9, samples_per_fault)
                
                for i in range(samples_per_fault):
                    sample_data = {k: v[i] for k, v in data.items()}
                    
                    if fault_type == 'Refrigerant_Leak':
                        sample_data = self.inject_refrigerant_leak(sample_data, severity[i])
                    elif fault_type == 'Sensor_Drift':
                        sample_data = self.inject_sensor_drift(sample_data, severity[i])
                    elif fault_type == 'Compressor_Failure':
                        sample_data = self.inject_compressor_failure(sample_data, severity[i])
                    elif fault_type == 'Air_Filter_Clog':
                        sample_data = self.inject_air_filter_clog(sample_data, severity[i])
                    elif fault_type == 'Evaporator_Fouling':
                        sample_data = self.inject_evaporator_fouling(sample_data, severity[i])
                    elif fault_type == 'Condenser_Fouling':
                        sample_data = self.inject_condenser_fouling(sample_data, severity[i])
                    elif fault_type == 'Expansion_Valve_Stuck':
                        sample_data = self.inject_expansion_valve_stuck(sample_data, severity[i])
                    elif fault_type == 'Fan_Motor_Issue':
                        sample_data = self.inject_fan_motor_issue(sample_data, severity[i])
                    
                    for k, v in sample_data.items():
                        data[k][i] = v
            
            # Create DataFrame
            df = pd.DataFrame(data)
            df['fault_type'] = fault_type
            df['timestamp'] = [start_time + timedelta(hours=fault_idx * samples_per_fault + i) 
                              for i in range(samples_per_fault)]
            
            all_data.append(df)
        
        # Combine and shuffle
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nDataset generated: {len(final_df)} samples")
        print(f"Fault distribution:\n{final_df['fault_type'].value_counts()}")
        
        return final_df

if __name__ == "__main__":
    generator = HVACDataGenerator()
    df = generator.generate_dataset(hours=25000, samples_per_hour=1)
    
    # Save dataset
    df.to_csv('hvac_data.csv', index=False)
    print("\nDataset saved to 'hvac_data.csv'")
