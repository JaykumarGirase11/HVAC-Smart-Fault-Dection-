# 🚀 Quick Start Guide

## Run the Dashboard

Open your terminal and run:

```bash
./start_dashboard.sh
```

OR

```bash
./venv/bin/streamlit run app.py
```

The dashboard will open automatically at: **http://localhost:8501**

## What You'll See

### 1. Professional Dashboard
- Siemens-branded interface
- Animated UI elements
- Real-time monitoring capabilities

### 2. Features to Demo
- **Real-Time Monitoring**: Test different fault scenarios
- **Historical Analysis**: Upload CSV data
- **System Training**: View model performance
- **Alert Management**: See automated diagnostics

## Demo Scenarios

### Scenario 1: Refrigerant Leak (Critical)
```
Supply Air Temp: 68°F
Refrigerant Pressure: 95 PSI (Low!)
Power Consumption: 14.5 kW (High!)
```
Expected: Critical alert with immediate action required

### Scenario 2: Air Filter Clog (Medium)
```
Airflow Rate: 1500 CFM (Low!)
Power Consumption: 11 kW
Supply Air Temp: 63°F
```
Expected: Medium severity, schedule maintenance

### Scenario 3: Normal Operation
```
Supply Air Temp: 60°F
Refrigerant Pressure: 130 PSI
Airflow Rate: 2000 CFM
Power Consumption: 10 kW
```
Expected: System normal, no action needed

## Project Files

- ✅ `hvac_data.csv` - 25,000 hours of training data
- ✅ `hvac_model_ensemble.pkl` - Trained AI model (80% accuracy)
- ✅ `confusion_matrix.png` - Model performance visualization
- ✅ `feature_importance.png` - Feature analysis

## For Siemens Interview

Show them:
1. The professional dashboard design
2. Real-time fault detection
3. Model accuracy (80%)
4. Automated alert system
5. Your understanding of HVAC systems

**You're ready to impress Siemens! 🎯**
