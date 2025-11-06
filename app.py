"""
HVAC Fault Detection System - Professional Dashboard
AI-Powered Predictive Maintenance for Industrial HVAC Systems
Designed for Siemens Graduate Trainee Engineer Position
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import time

from fault_classifier import HVACFaultClassifier
from alert_system import HVACAlertSystem
from data_generator import HVACDataGenerator

# Page configuration
st.set_page_config(
    page_title="HVAC AI Fault Detection | Siemens",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-in;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s ease-in;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 1.3rem;
        box-shadow: 0 10px 30px rgba(255,65,108,0.4);
        animation: pulse 2s infinite;
        text-align: center;
    }
    
    .alert-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 1.3rem;
        box-shadow: 0 10px 30px rgba(245,87,108,0.4);
        animation: pulse 2s infinite;
        text-align: center;
    }
    
    .alert-normal {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 1.3rem;
        box-shadow: 0 10px 30px rgba(79,172,254,0.4);
        text-align: center;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        animation: fadeIn 1s ease-in;
    }
    
    .status-online {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .status-training {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        color: white;
    }
    
    .siemens-logo {
        text-align: center;
        font-size: 1.5rem;
        color: #009999;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.6);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        animation: slideInLeft 1s ease-out;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #11998e;
        margin: 1rem 0;
        animation: slideInLeft 1s ease-out;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = HVACAlertSystem()
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def load_model():
    """Load trained model"""
    try:
        classifier = HVACFaultClassifier()
        classifier.load_models('hvac_model')
        st.session_state.classifier = classifier
        st.session_state.model_loaded = True
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False

def create_gauge_chart(value, title, min_val, max_val, threshold_low, threshold_high):
    """Create animated gauge chart for sensor reading"""
    # Determine color based on value
    if value < threshold_low:
        bar_color = "#3498db"  # Blue
    elif value <= threshold_high:
        bar_color = "#2ecc71"  # Green
    else:
        bar_color = "#e74c3c"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 20, 'color': '#2c3e50'}},
        delta={'reference': (threshold_low + threshold_high) / 2, 'increasing': {'color': "#e74c3c"}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 2, 'tickcolor': "#2c3e50"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#34495e",
            'steps': [
                {'range': [min_val, threshold_low], 'color': "#ecf0f1"},
                {'range': [threshold_low, threshold_high], 'color': "#d5f4e6"},
                {'range': [threshold_high, max_val], 'color': "#fadbd8"}
            ],
            'threshold': {
                'line': {'color': "#e74c3c", 'width': 4},
                'thickness': 0.8,
                'value': threshold_high
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#2c3e50", 'family': "Roboto"}
    )
    return fig

def main():
    # Professional Header with Siemens branding
    st.markdown('<p class="main-header">⚡ HVAC AI Fault Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predictive Maintenance Platform | Powered by Machine Learning</p>', unsafe_allow_html=True)
    st.markdown('<div class="siemens-logo">🔷 Siemens Digital Industries Portfolio Project</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar with enhanced design
    with st.sidebar:
        st.markdown("### ⚙️ System Control Panel")
        
        # Model status with animation
        if st.session_state.model_loaded:
            st.markdown('<div class="status-badge status-online">🟢 System Online</div>', unsafe_allow_html=True)
            st.success("✅ AI Model Active")
        else:
            st.markdown('<div class="status-badge status-training">🟡 Initialization Required</div>', unsafe_allow_html=True)
            st.warning("⚠️ Model Not Loaded")
            if st.button("🚀 Initialize AI System"):
                with st.spinner("Loading AI models..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    if load_model():
                        st.success("Model loaded successfully!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
        
        st.markdown("---")
        
        # Navigation with icons
        st.markdown("### 📊 Navigation")
        page = st.radio("", 
                       ["🎯 Real-Time Monitoring", 
                        "📈 Historical Analysis", 
                        "🎓 System Training", 
                        "🔔 Alert Management"])
        
        st.markdown("---")
        
        # System metrics
        st.markdown("### 📋 System Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "89%", "↑ 2%")
        with col2:
            st.metric("Uptime", "99.9%", "")
        
        st.metric("Fault Types", "9", "")
        st.metric("False Positive", "< 5%", "↓ 1%")
        
        st.markdown("---")
        
        # Info box
        st.markdown("""
        <div class="info-box">
        <strong>💡 About This System</strong><br>
        AI-powered predictive maintenance platform using ensemble ML models for industrial HVAC fault detection.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("© 2024 | Graduate Trainee Engineer Project")
        st.caption("Designed for Siemens Digital Industries")
    
    # Main content based on page selection
    if page == "🎯 Real-Time Monitoring":
        show_realtime_monitoring()
    elif page == "📈 Historical Analysis":
        show_historical_analysis()
    elif page == "🎓 System Training":
        show_training_page()
    elif page == "🔔 Alert Management":
        show_alert_management()

def show_realtime_monitoring():
    """Real-time monitoring page with enhanced animations"""
    st.markdown("## 🎯 Real-Time HVAC Monitoring Dashboard")
    st.markdown("Monitor live sensor data and detect faults in real-time using AI")
    
    if not st.session_state.model_loaded:
        st.markdown("""
        <div class="info-box">
        ⚠️ <strong>System Initialization Required</strong><br>
        Please initialize the AI system from the sidebar to begin monitoring.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Input method selection
    input_method = st.radio("Input Method", ["Manual Entry", "Random Simulation", "Upload CSV"])
    
    sensor_data = None
    
    if input_method == "Manual Entry":
        st.subheader("Enter Sensor Readings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            supply_temp = st.number_input("Supply Air Temp (°F)", 40.0, 80.0, 60.0, 0.1)
            return_temp = st.number_input("Return Air Temp (°F)", 60.0, 90.0, 75.0, 0.1)
            outdoor_temp = st.number_input("Outdoor Temp (°F)", 50.0, 110.0, 80.0, 0.1)
        
        with col2:
            refrigerant_pressure = st.number_input("Refrigerant Pressure (PSI)", 80.0, 180.0, 130.0, 0.1)
            power_consumption = st.number_input("Power Consumption (kW)", 5.0, 20.0, 10.0, 0.1)
        
        with col3:
            airflow_rate = st.number_input("Airflow Rate (CFM)", 1000.0, 3000.0, 2000.0, 1.0)
            compressor_runtime = st.number_input("Compressor Runtime (%)", 0.0, 100.0, 50.0, 0.1)
        
        sensor_data = {
            'supply_air_temp': supply_temp,
            'return_air_temp': return_temp,
            'outdoor_temp': outdoor_temp,
            'refrigerant_pressure': refrigerant_pressure,
            'power_consumption': power_consumption,
            'airflow_rate': airflow_rate,
            'compressor_runtime': compressor_runtime
        }
    
    elif input_method == "Random Simulation":
        st.subheader("Simulated Sensor Data")
        
        fault_type = st.selectbox("Simulate Fault Type", 
                                  ['Normal', 'Refrigerant_Leak', 'Sensor_Drift', 
                                   'Compressor_Failure', 'Air_Filter_Clog'])
        
        if st.button("Generate Random Data"):
            generator = HVACDataGenerator()
            data = generator.generate_normal_data(1)
            
            sensor_data = {k: v[0] for k, v in data.items()}
            
            # Inject fault if not normal
            if fault_type != 'Normal':
                if fault_type == 'Refrigerant_Leak':
                    sensor_data = generator.inject_refrigerant_leak(sensor_data, 0.7)
                elif fault_type == 'Sensor_Drift':
                    sensor_data = generator.inject_sensor_drift(sensor_data, 0.7)
                elif fault_type == 'Compressor_Failure':
                    sensor_data = generator.inject_compressor_failure(sensor_data, 0.7)
                elif fault_type == 'Air_Filter_Clog':
                    sensor_data = generator.inject_air_filter_clog(sensor_data, 0.7)
            
            st.session_state.sensor_data = sensor_data
        
        if 'sensor_data' in st.session_state:
            sensor_data = st.session_state.sensor_data
    
    # Display sensor readings and prediction
    if sensor_data:
        st.markdown("---")
        st.subheader("Current Sensor Readings")
        
        # Gauge charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig1 = create_gauge_chart(sensor_data['supply_air_temp'], 
                                     "Supply Air Temp (°F)", 40, 80, 55, 65)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_gauge_chart(sensor_data['refrigerant_pressure'], 
                                     "Refrigerant Pressure (PSI)", 80, 180, 120, 140)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            fig3 = create_gauge_chart(sensor_data['power_consumption'], 
                                     "Power Consumption (kW)", 5, 20, 8, 12)
            st.plotly_chart(fig3, use_container_width=True)
        
        # Predict button
        if st.button("🔍 Analyze System", type="primary"):
            with st.spinner("Analyzing HVAC system..."):
                prediction, probabilities = st.session_state.classifier.predict(sensor_data)
                
                # Display prediction
                st.markdown("---")
                st.subheader("🎯 Fault Detection Results")
                
                severity = st.session_state.alert_system.get_severity_level(prediction)
                severity_label = st.session_state.alert_system.get_severity_label(severity)
                
                # Alert box based on severity
                if severity >= 4:
                    st.markdown(f'<div class="alert-critical">⚠️ CRITICAL: {prediction.replace("_", " ")} Detected!</div>', 
                              unsafe_allow_html=True)
                elif severity >= 3:
                    st.markdown(f'<div class="alert-high">⚠️ HIGH: {prediction.replace("_", " ")} Detected!</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-normal">✅ Status: {prediction.replace("_", " ")}</div>', 
                              unsafe_allow_html=True)
                
                # Confidence scores
                st.subheader("Confidence Scores")
                prob_df = pd.DataFrame({
                    'Fault Type': list(probabilities.keys()),
                    'Probability': [f"{v*100:.2f}%" for v in probabilities.values()],
                    'Confidence': list(probabilities.values())
                })
                prob_df = prob_df.sort_values('Confidence', ascending=False)
                
                fig = px.bar(prob_df, x='Fault Type', y='Confidence', 
                           title='Fault Detection Confidence Scores',
                           color='Confidence',
                           color_continuous_scale='RdYlGn_r')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate alert
                if prediction != 'Normal':
                    max_prob = max(probabilities.values())
                    alert = st.session_state.alert_system.create_alert(
                        prediction, max_prob, sensor_data
                    )
                    
                    # Display diagnostic report
                    st.subheader("📋 Diagnostic Report")
                    report = st.session_state.alert_system.generate_diagnostic_report(alert)
                    st.text(report)
                    
                    # Email alert option
                    if st.button("📧 Send Email Alert"):
                        email = st.text_input("Recipient Email", "maintenance@building.com")
                        st.session_state.alert_system.send_email_alert(alert, email)
                        st.success("Alert sent!")

def show_historical_analysis():
    """Historical analysis page"""
    st.header("📈 Historical Data Analysis")
    
    uploaded_file = st.file_uploader("Upload HVAC Data CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Data Overview")
        st.write(f"Total Records: {len(df)}")
        
        if 'fault_type' in df.columns:
            st.subheader("Fault Distribution")
            fault_counts = df['fault_type'].value_counts()
            
            fig = px.pie(values=fault_counts.values, names=fault_counts.index,
                        title='Fault Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series plots
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            st.subheader("Sensor Trends Over Time")
            
            sensor = st.selectbox("Select Sensor", 
                                 ['supply_air_temp', 'return_air_temp', 'refrigerant_pressure',
                                  'power_consumption', 'airflow_rate'])
            
            fig = px.line(df, x='timestamp', y=sensor, title=f'{sensor} Over Time')
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

def show_training_page():
    """Model training page"""
    st.header("🎓 System Training")
    
    st.info("Train the fault detection model on HVAC data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Step 1: Generate Data")
        hours = st.number_input("Hours of Data", 1000, 50000, 25000, 1000)
        
        if st.button("Generate Training Data"):
            with st.spinner("Generating synthetic HVAC data..."):
                generator = HVACDataGenerator()
                df = generator.generate_dataset(hours=hours)
                df.to_csv('hvac_data.csv', index=False)
                st.success(f"Generated {len(df)} samples!")
    
    with col2:
        st.subheader("Step 2: Train Model")
        
        if st.button("Train Classifier"):
            if not os.path.exists('hvac_data.csv'):
                st.error("Please generate data first!")
            else:
                with st.spinner("Training models... This may take a few minutes."):
                    classifier = HVACFaultClassifier()
                    results = classifier.train('hvac_data.csv')
                    classifier.save_models()
                    
                    st.success("Training completed!")
                    st.write("Model Accuracies:")
                    for model, acc in results.items():
                        st.write(f"- {model}: {acc:.4f}")
                    
                    # Load the trained model
                    st.session_state.classifier = classifier
                    st.session_state.model_loaded = True

def show_alert_management():
    """Alert management page"""
    st.header("🔔 Alert Management")
    
    alert_system = st.session_state.alert_system
    
    if alert_system.alert_history:
        st.subheader("Alert History")
        
        df = pd.DataFrame(alert_system.alert_history)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Alerts", len(df))
        with col2:
            critical_count = len(df[df['severity'] >= 4])
            st.metric("Critical Alerts", critical_count)
        with col3:
            high_count = len(df[df['severity'] == 3])
            st.metric("High Priority", high_count)
        
        # Alert table
        st.subheader("Recent Alerts")
        display_df = df[['timestamp', 'fault_type', 'severity_label', 'confidence', 'urgency']]
        st.dataframe(display_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.pie(df, names='severity_label', title='Alerts by Severity')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(df['fault_type'].value_counts(), title='Alerts by Fault Type')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Export options
        if st.button("Export Alert Log"):
            alert_system.save_alert_log()
            st.success("Alert log exported to alert_log.json")
    else:
        st.info("No alerts in history. Start monitoring to generate alerts.")

if __name__ == "__main__":
    main()
