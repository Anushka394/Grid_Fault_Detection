import pandas as pd
import json
import numpy as np
from datetime import datetime
from alert_manager import AlertManager

class EnhancedFaultDetector:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.thresholds = self.config['fault_thresholds']
        self.duration_threshold = self.config.get('fault_duration_threshold', 1)
        self.alert_manager = AlertManager(config_path)
        
        # Initialize fault counters for each grid section
        self.fault_counters = {}
        self.fault_history = []
        
    def detect_faults(self, data, grid_section='Substation_A'):
        """Enhanced fault detection with multiple fault types and alerting"""
        faults = []
        
        # Initialize counters for this grid section if not exists
        if grid_section not in self.fault_counters:
            self.fault_counters[grid_section] = {
                "under_voltage": 0, "over_current": 0, "under_frequency": 0,
                "over_frequency": 0, "low_power_factor": 0, "voltage_sag": 0
            }
        
        counters = self.fault_counters[grid_section]
        th = self.thresholds

        for _, row in data.iterrows():
            fault_found_in_row = False
            
            # Earth fault detection (critical)
            if (row['Voltage(V)'] < th['earth_fault']['max_voltage'] and 
                row['Current(A)'] > th['earth_fault']['min_current']):
                
                fault_data = {
                    'timestamp': row['Timestamp'],
                    'fault_type': 'Earth Fault',
                    'parameter': 'Voltage/Current',
                    'value': f"{row['Voltage(V)']}V / {row['Current(A)']}A",
                    'grid_section': grid_section
                }
                
                # Process through alert manager
                processed_fault = self.alert_manager.process_fault(fault_data)
                faults.append((
                    processed_fault['timestamp'],
                    processed_fault['fault_type'],
                    processed_fault['parameter'],
                    processed_fault['value'],
                    processed_fault['severity']
                ))
                fault_found_in_row = True

            # Reset all counters if a critical fault is found
            if fault_found_in_row:
                for key in counters:
                    counters[key] = 0
                continue

            # Check other fault conditions
            fault_checks = [
                ('under_voltage', row['Voltage(V)'] < th['under_voltage']['max_voltage'], 
                 'Voltage', f"{row['Voltage(V)']}V"),
                ('over_current', row['Current(A)'] > th['over_current']['min_current'],
                 'Current', f"{row['Current(A)']}A"),
                ('under_frequency', row['Frequency(Hz)'] < th['under_frequency']['min_freq'],
                 'Frequency', f"{row['Frequency(Hz)']}Hz"),
                ('over_frequency', row['Frequency(Hz)'] > th['over_frequency']['max_freq'],
                 'Frequency', f"{row['Frequency(Hz)']}Hz"),
                ('low_power_factor', row['PowerFactor'] < th['low_power_factor']['min_pf'],
                 'Power Factor', f"{row['PowerFactor']}"),
                ('voltage_sag', th['voltage_sag']['min_voltage'] <= row['Voltage(V)'] <= th['voltage_sag']['max_voltage'],
                 'Voltage', f"{row['Voltage(V)']}V")
            ]
            
            for fault_type, condition, param, value in fault_checks:
                if condition:
                    counters[fault_type] += 1
                else:
                    counters[fault_type] = 0
                
                # Trigger fault if duration threshold is met
                if counters[fault_type] == self.duration_threshold:
                    fault_name = self._get_fault_display_name(fault_type)
                    
                    fault_data = {
                        'timestamp': row['Timestamp'],
                        'fault_type': fault_name,
                        'parameter': param,
                        'value': value,
                        'grid_section': grid_section
                    }
                    
                    processed_fault = self.alert_manager.process_fault(fault_data)
                    faults.append((
                        processed_fault['timestamp'],
                        processed_fault['fault_type'],
                        processed_fault['parameter'],
                        processed_fault['value'],
                        processed_fault['severity']
                    ))

        return faults
    
    def _get_fault_display_name(self, fault_type):
        """Convert internal fault type to display name"""
        display_names = {
            'under_voltage': 'Under-voltage',
            'over_current': 'Overcurrent',
            'under_frequency': 'Under-frequency',
            'over_frequency': 'Over-frequency',
            'low_power_factor': 'Low Power Factor',
            'voltage_sag': 'Voltage Sag'
        }
        return display_names.get(fault_type, fault_type.title())
    
    def detect_harmonics(self, data):
        """Detect harmonic distortion (simulated)"""
        harmonics_faults = []
        
        # Simulate THD calculation based on current variations
        for _, row in data.iterrows():
            # Simple harmonic detection based on current fluctuation
            if hasattr(self, 'prev_current'):
                current_change = abs(row['Current(A)'] - self.prev_current)
                if current_change > 2.0:  # Significant current change might indicate harmonics
                    thd_estimate = current_change * 2.5  # Simplified THD estimation
                    
                    if thd_estimate > self.thresholds['harmonics']['max_thd']:
                        fault_data = {
                            'timestamp': row['Timestamp'],
                            'fault_type': 'Harmonic Distortion',
                            'parameter': 'THD',
                            'value': f"{thd_estimate:.2f}%",
                            'grid_section': 'Substation_A'
                        }
                        
                        processed_fault = self.alert_manager.process_fault(fault_data)
                        harmonics_faults.append((
                            processed_fault['timestamp'],
                            processed_fault['fault_type'],
                            processed_fault['parameter'],
                            processed_fault['value'],
                            processed_fault['severity']
                        ))
            
            self.prev_current = row['Current(A)']
        
        return harmonics_faults
    
    def predict_potential_faults(self, data):
        """Simple predictive analysis for potential faults"""
        predictions = []
        
        if len(data) < 5:
            return predictions
        
        # Analyze trends in last 5 readings
        recent_data = data.tail(5)
        
        # Voltage trend analysis
        voltage_trend = np.polyfit(range(len(recent_data)), recent_data['Voltage(V)'], 1)[0]
        if voltage_trend < -2:  # Declining voltage
            predictions.append({
                'type': 'Potential Under-voltage',
                'confidence': min(abs(voltage_trend) * 10, 95),
                'estimated_time': '5-10 minutes',
                'parameter': 'Voltage'
            })
        
        # Current trend analysis
        current_trend = np.polyfit(range(len(recent_data)), recent_data['Current(A)'], 1)[0]
        if current_trend > 1:  # Rising current
            predictions.append({
                'type': 'Potential Overcurrent',
                'confidence': min(current_trend * 15, 90),
                'estimated_time': '3-8 minutes',
                'parameter': 'Current'
            })
        
        # Frequency stability analysis
        freq_std = recent_data['Frequency(Hz)'].std()
        if freq_std > 0.3:  # High frequency variation
            predictions.append({
                'type': 'Frequency Instability',
                'confidence': min(freq_std * 100, 85),
                'estimated_time': '2-5 minutes',
                'parameter': 'Frequency'
            })
        
        return predictions
    
    def get_fault_statistics(self):
        """Get fault statistics from alert manager"""
        return self.alert_manager.get_alert_statistics()
    
    def get_recent_faults(self, limit=20):
        """Get recent faults from database"""
        return self.alert_manager.get_recent_alerts(limit)
    
    def cleanup_old_data(self):
        """Clean up old fault data"""
        return self.alert_manager.cleanup_old_alerts()