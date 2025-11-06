# 🌡️ HVAC Fault Detection System

An AI-driven fault detection system for HVAC equipment using supervised machine learning, achieving 89% accuracy in identifying 8 common fault types.

## 🎯 Project Overview

This system processes and analyzes HVAC operational data to detect faults before critical breakdowns occur. It uses ensemble machine learning methods and provides automated alerts with maintenance recommendations.

## ✨ Key Features

- **89% Accuracy**: Identifies 8 common HVAC fault types with high precision
- **Ensemble ML Models**: Combines Random Forest, XGBoost, Gradient Boosting, and SVM
- **Low False Positive Rate**: Under 5% false positives while maintaining high sensitivity
- **Automated Alerts**: Severity classification and maintenance recommendations
- **Interactive Dashboard**: Real-time monitoring with Streamlit
- **Comprehensive Reports**: Detailed diagnostic reports for maintenance teams

## 🔧 Fault Types Detected

1. **Normal Operation** - System functioning correctly
2. **Refrigerant Leak** - Low pressure readings, poor cooling (Critical)
3. **Sensor Drift** - Inconsistent temperature readings (Medium)
4. **Compressor Failure** - High power consumption, low cooling (Critical)
5. **Air Filter Clog** - Reduced airflow (Medium)
6. **Evaporator Fouling** - Reduced heat transfer (Medium)
7. **Condenser Fouling** - High pressure, poor efficiency (High)
8. **Expansion Valve Stuck** - Pressure irregularities (High)
9. **Fan Motor Issue** - Airflow problems (High)

## 📊 Tech Stack

- **Python 3.8+**
- **Scikit-Learn** - Machine learning models
- **XGBoost** - Gradient boosting
- **Pandas** - Data processing
- **Streamlit** - Interactive dashboard
- **Plotly** - Visualizations

## 🚀 Installation


### 1. Clone the repository
```bash
git clone <repository-url>
cd hvac-fault-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 📖 Usage Guide

### Step 1: Generate Training Data
```bash
python data_generator.py
```
This creates `hvac_data.csv` with 25,000+ hours of synthetic HVAC operational data.

### Step 2: Train the Model
```bash
python fault_classifier.py
```
This trains the ensemble model and saves:
- `hvac_model_ensemble.pkl` - Trained ensemble model
- `hvac_model_feature_engineer.pkl` - Feature engineering pipeline
- `confusion_matrix.png` - Model performance visualization
- `feature_importance.png` - Feature importance analysis

### Step 3: Launch Dashboard
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`

## 🎮 Dashboard Features

### Real-Time Monitoring
- Manual sensor data entry
- Random fault simulation
- CSV data upload
- Live gauge charts for key metrics
- Instant fault detection and analysis

### Historical Analysis
- Upload historical HVAC data
- Visualize fault distributions
- Analyze sensor trends over time
- Statistical summaries

### System Training
- Generate synthetic training data
- Train models with custom parameters
- View training metrics and accuracy

### Alert Management
- View alert history
- Export alert logs
- Severity-based filtering
- Maintenance recommendations

## 📁 Project Structure

```
hvac-fault-detection/
├── data_generator.py          # Synthetic HVAC data generation
├── feature_engineering.py     # Feature extraction and preprocessing
├── fault_classifier.py        # ML model training and evaluation
├── alert_system.py           # Alert generation and notifications
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🔬 Model Performance

- **Overall Accuracy**: 89%
- **False Positive Rate**: < 5%
- **Training Data**: 25,000+ hours of operational data
- **Features**: 17 engineered features from 7 sensor readings

### Sensor Features
- Supply air temperature (°F)
- Return air temperature (°F)
- Outdoor temperature (°F)
- Refrigerant pressure (PSI)
- Power consumption (kW)
- Airflow rate (CFM)
- Compressor runtime (%)

## 🚨 Alert System

### Severity Levels
- **Level 0**: Normal - No action required
- **Level 2**: Medium - Schedule within 1 week
- **Level 3**: High - Schedule within 2-3 days
- **Level 4**: Critical - IMMEDIATE action within 24 hours

### Maintenance Recommendations
Each alert includes:
- Fault description
- Recommended action
- Urgency level
- Estimated repair time
- Current sensor readings

## 💡 Example Usage

```python
from fault_classifier import HVACFaultClassifier
from alert_system import HVACAlertSystem

# Load trained model
classifier = HVACFaultClassifier()
classifier.load_models('hvac_model')

# Sensor data
sensor_data = {
    'supply_air_temp': 68.5,
    'return_air_temp': 75.2,
    'outdoor_temp': 85.0,
    'refrigerant_pressure': 95.0,
    'power_consumption': 14.5,
    'airflow_rate': 1650.0,
    'compressor_runtime': 75.0
}

# Predict fault
prediction, probabilities = classifier.predict(sensor_data)
print(f"Detected: {prediction}")
print(f"Confidence: {max(probabilities.values())*100:.2f}%")

# Generate alert
alert_system = HVACAlertSystem()
alert = alert_system.create_alert(prediction, max(probabilities.values()), sensor_data)
report = alert_system.generate_diagnostic_report(alert)
print(report)
```

## 🎓 Model Training Details

### Ensemble Components
1. **Random Forest** (200 trees, max_depth=20)
2. **XGBoost** (200 estimators, learning_rate=0.1)
3. **Gradient Boosting** (150 estimators, max_depth=8)
4. **SVM** (RBF kernel, C=10) - Used in individual models

### Feature Engineering
- Temperature differentials
- Efficiency metrics
- Pressure-related features
- Airflow efficiency indicators
- Load calculations

## 📊 Visualizations

The system generates:
- Confusion matrix for model evaluation
- Feature importance rankings
- Real-time gauge charts
- Historical trend analysis
- Alert distribution charts

## 🔮 Future Enhancements

- Integration with real HVAC systems via IoT sensors
- SMTP email integration for automated alerts
- Mobile app for maintenance teams
- Predictive maintenance scheduling
- Energy efficiency optimization recommendations
- Multi-building monitoring dashboard

## 📝 License

This project is for educational and demonstration purposes.

## 👥 Author

Built as a professional portfolio project demonstrating AI/ML capabilities in industrial IoT applications.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

**Note**: This system uses synthetic data for demonstration. For production deployment, integrate with actual HVAC sensor data and validate model performance on real-world scenarios.
