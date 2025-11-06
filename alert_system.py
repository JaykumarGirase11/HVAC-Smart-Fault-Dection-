"""
HVAC Alert System
Generates alerts, severity classification, and maintenance recommendations
"""

import pandas as pd
from datetime import datetime
import json

class HVACAlertSystem:
    def __init__(self):
        # Fault severity levels (1=Low, 2=Medium, 3=High, 4=Critical)
        self.fault_severity = {
            'Normal': 0,
            'Sensor_Drift': 2,
            'Air_Filter_Clog': 2,
            'Evaporator_Fouling': 2,
            'Fan_Motor_Issue': 3,
            'Condenser_Fouling': 3,
            'Expansion_Valve_Stuck': 3,
            'Refrigerant_Leak': 4,
            'Compressor_Failure': 4
        }
        
        # Maintenance recommendations
        self.maintenance_actions = {
            'Normal': {
                'action': 'No action required',
                'description': 'System operating normally',
                'urgency': 'None',
                'estimated_time': '0 hours'
            },
            'Sensor_Drift': {
                'action': 'Calibrate or replace temperature sensors',
                'description': 'Temperature sensors showing inconsistent readings. Calibration or replacement needed.',
                'urgency': 'Schedule within 1 week',
                'estimated_time': '1-2 hours'
            },
            'Air_Filter_Clog': {
                'action': 'Replace air filters',
                'description': 'Reduced airflow detected. Replace air filters immediately to prevent further issues.',
                'urgency': 'Schedule within 2-3 days',
                'estimated_time': '0.5-1 hour'
            },
            'Evaporator_Fouling': {
                'action': 'Clean evaporator coils',
                'description': 'Evaporator coils require cleaning. Reduced heat transfer efficiency detected.',
                'urgency': 'Schedule within 1 week',
                'estimated_time': '2-3 hours'
            },
            'Fan_Motor_Issue': {
                'action': 'Inspect and repair/replace fan motor',
                'description': 'Fan motor showing signs of failure. Inspect bearings, belts, and electrical connections.',
                'urgency': 'Schedule within 2-3 days',
                'estimated_time': '2-4 hours'
            },
            'Condenser_Fouling': {
                'action': 'Clean condenser coils',
                'description': 'Condenser coils fouled. Clean outdoor unit and check for debris blockage.',
                'urgency': 'Schedule within 3-5 days',
                'estimated_time': '2-3 hours'
            },
            'Expansion_Valve_Stuck': {
                'action': 'Inspect and replace expansion valve',
                'description': 'Expansion valve malfunction detected. May require replacement.',
                'urgency': 'Schedule within 2-3 days',
                'estimated_time': '3-4 hours'
            },
            'Refrigerant_Leak': {
                'action': 'URGENT: Locate and repair refrigerant leak, recharge system',
                'description': 'Critical refrigerant leak detected. System efficiency severely compromised. Immediate attention required.',
                'urgency': 'IMMEDIATE - Schedule within 24 hours',
                'estimated_time': '4-6 hours'
            },
            'Compressor_Failure': {
                'action': 'URGENT: Inspect compressor, likely replacement needed',
                'description': 'Compressor failure imminent or in progress. System may shut down. Immediate inspection required.',
                'urgency': 'IMMEDIATE - Schedule within 24 hours',
                'estimated_time': '6-8 hours'
            }
        }
        
        self.alert_history = []
    
    def get_severity_level(self, fault_type):
        """Get severity level for a fault"""
        return self.fault_severity.get(fault_type, 0)
    
    def get_severity_label(self, severity):
        """Convert severity number to label"""
        labels = {0: 'Normal', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}
        return labels.get(severity, 'Unknown')
    
    def create_alert(self, fault_type, confidence, sensor_data=None):
        """Create alert for detected fault"""
        severity = self.get_severity_level(fault_type)
        severity_label = self.get_severity_label(severity)
        maintenance = self.maintenance_actions.get(fault_type, {})
        
        alert = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fault_type': fault_type,
            'confidence': f"{confidence * 100:.2f}%",
            'severity': severity,
            'severity_label': severity_label,
            'maintenance_action': maintenance.get('action', 'Unknown'),
            'description': maintenance.get('description', 'No description available'),
            'urgency': maintenance.get('urgency', 'Unknown'),
            'estimated_repair_time': maintenance.get('estimated_time', 'Unknown'),
            'sensor_data': sensor_data
        }
        
        # Store in history
        self.alert_history.append(alert)
        
        return alert
    
    def generate_diagnostic_report(self, alert):
        """Generate detailed diagnostic report"""
        report = f"""
{'='*70}
HVAC FAULT DETECTION DIAGNOSTIC REPORT
{'='*70}

ALERT INFORMATION:
------------------
Timestamp:          {alert['timestamp']}
Fault Detected:     {alert['fault_type'].replace('_', ' ')}
Confidence Level:   {alert['confidence']}
Severity:           {alert['severity_label']} (Level {alert['severity']}/4)

MAINTENANCE RECOMMENDATION:
---------------------------
Action Required:    {alert['maintenance_action']}
Description:        {alert['description']}
Urgency:            {alert['urgency']}
Estimated Time:     {alert['estimated_repair_time']}

"""
        
        if alert['sensor_data']:
            report += f"""CURRENT SENSOR READINGS:
------------------------
Supply Air Temp:        {alert['sensor_data'].get('supply_air_temp', 'N/A'):.1f} °F
Return Air Temp:        {alert['sensor_data'].get('return_air_temp', 'N/A'):.1f} °F
Outdoor Temp:           {alert['sensor_data'].get('outdoor_temp', 'N/A'):.1f} °F
Refrigerant Pressure:   {alert['sensor_data'].get('refrigerant_pressure', 'N/A'):.1f} PSI
Power Consumption:      {alert['sensor_data'].get('power_consumption', 'N/A'):.1f} kW
Airflow Rate:           {alert['sensor_data'].get('airflow_rate', 'N/A'):.1f} CFM
Compressor Runtime:     {alert['sensor_data'].get('compressor_runtime', 'N/A'):.1f}%

"""
        
        report += f"""RECOMMENDED ACTIONS:
--------------------
1. Review sensor readings and compare with normal operating ranges
2. {alert['maintenance_action']}
3. Document all maintenance activities
4. Monitor system performance after repairs
5. Schedule follow-up inspection if issues persist

{'='*70}
"""
        
        return report
    
    def should_send_alert(self, fault_type, confidence_threshold=0.7):
        """Determine if alert should be sent based on severity and confidence"""
        severity = self.get_severity_level(fault_type)
        
        # Always alert for critical faults
        if severity >= 4:
            return True
        
        # Alert for high severity with good confidence
        if severity >= 3:
            return True
        
        # Alert for medium severity with high confidence
        if severity >= 2:
            return True
        
        return False
    
    def send_email_alert(self, alert, recipient_email):
        """
        Simulate sending email alert
        In production, integrate with SMTP or email service
        """
        email_content = {
            'to': recipient_email,
            'subject': f"HVAC Alert: {alert['fault_type'].replace('_', ' ')} - {alert['severity_label']} Severity",
            'body': self.generate_diagnostic_report(alert)
        }
        
        print(f"\n[EMAIL ALERT SENT]")
        print(f"To: {email_content['to']}")
        print(f"Subject: {email_content['subject']}")
        print("\n--- Email Body ---")
        print(email_content['body'])
        
        return email_content
    
    def get_alert_summary(self):
        """Get summary of all alerts"""
        if not self.alert_history:
            return "No alerts in history"
        
        df = pd.DataFrame(self.alert_history)
        
        summary = f"""
Alert Summary:
--------------
Total Alerts: {len(self.alert_history)}

By Severity:
{df['severity_label'].value_counts().to_string()}

By Fault Type:
{df['fault_type'].value_counts().to_string()}
"""
        return summary
    
    def save_alert_log(self, filename='alert_log.json'):
        """Save alert history to file"""
        with open(filename, 'w') as f:
            json.dump(self.alert_history, f, indent=2)
        print(f"\nAlert log saved to '{filename}'")
    
    def load_alert_log(self, filename='alert_log.json'):
        """Load alert history from file"""
        try:
            with open(filename, 'r') as f:
                self.alert_history = json.load(f)
            print(f"Alert log loaded from '{filename}'")
        except FileNotFoundError:
            print(f"No alert log found at '{filename}'")

if __name__ == "__main__":
    # Test alert system
    alert_system = HVACAlertSystem()
    
    # Simulate fault detection
    test_sensor_data = {
        'supply_air_temp': 68.5,
        'return_air_temp': 75.2,
        'outdoor_temp': 85.0,
        'refrigerant_pressure': 95.0,
        'power_consumption': 14.5,
        'airflow_rate': 1650.0,
        'compressor_runtime': 75.0
    }
    
    # Create alert for refrigerant leak
    alert = alert_system.create_alert('Refrigerant_Leak', 0.92, test_sensor_data)
    
    # Generate report
    report = alert_system.generate_diagnostic_report(alert)
    print(report)
    
    # Simulate email
    alert_system.send_email_alert(alert, 'maintenance@building.com')
    
    # Save log
    alert_system.save_alert_log()
