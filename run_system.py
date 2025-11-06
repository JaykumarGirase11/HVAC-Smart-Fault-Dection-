"""
Quick Start Script for HVAC Fault Detection System
Automates the complete workflow: data generation, training, and dashboard launch
"""

import os
import sys
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"▶ {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print_header("HVAC Fault Detection System - Quick Start")
    
    # Check if data exists
    if not os.path.exists('hvac_data.csv'):
        print("📊 Step 1: Generating Training Data")
        print("This will create 25,000 hours of synthetic HVAC data...")
        if not run_command("python data_generator.py", "Data generation"):
            sys.exit(1)
    else:
        print("✓ Training data already exists (hvac_data.csv)\n")
    
    # Check if model exists
    if not os.path.exists('hvac_model_ensemble.pkl'):
        print("🎓 Step 2: Training Machine Learning Models")
        print("This may take a few minutes...")
        if not run_command("python fault_classifier.py", "Model training"):
            sys.exit(1)
    else:
        print("✓ Trained model already exists\n")
    
    # Launch dashboard
    print_header("🚀 Launching Streamlit Dashboard")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run("streamlit run app.py", shell=True)
    except KeyboardInterrupt:
        print("\n\n✓ Dashboard stopped. Thank you for using HVAC Fault Detection System!")

if __name__ == "__main__":
    main()
