import pandas as pd
import json

class FaultDetector:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.thresholds = config['fault_thresholds']
        self.duration_threshold = config.get('fault_duration_threshold', 1)

    def detect_faults(self, data):
        faults = []
        
        counters = {
            "under_voltage": 0, "over_current": 0, "under_frequency": 0,
            "over_frequency": 0, "low_power_factor": 0
        }
        
        th = self.thresholds

        for _, row in data.iterrows():
            fault_found_in_row = False

            if row['Voltage(V)'] < th['earth_fault']['max_voltage'] and row['Current(A)'] > th['earth_fault']['min_current']:
                faults.append((row['Timestamp'], "Possible Earth Fault", "Voltage/Current", f"{row['Voltage(V)']}V / {row['Current(A)']}A"))
                fault_found_in_row = True

            if fault_found_in_row:
                for key in counters: counters[key] = 0
                continue

            if row['Voltage(V)'] < th['under_voltage']['max_voltage']: counters['under_voltage'] += 1
            else: counters['under_voltage'] = 0
            
            if row['Current(A)'] > th['over_current']['min_current']: counters['over_current'] += 1
            else: counters['over_current'] = 0
            
            if row['Frequency(Hz)'] < th['under_frequency']['min_freq']: counters['under_frequency'] += 1
            else: counters['under_frequency'] = 0

            if row['Frequency(Hz)'] > th['over_frequency']['max_freq']: counters['over_frequency'] += 1
            else: counters['over_frequency'] = 0

            if row['PowerFactor'] < th['low_power_factor']['min_pf']: counters['low_power_factor'] += 1
            else: counters['low_power_factor'] = 0

            if counters['under_voltage'] == self.duration_threshold:
                faults.append((row['Timestamp'], "Under-voltage", "Voltage", f"{row['Voltage(V)']}V"))
            if counters['over_current'] == self.duration_threshold:
                faults.append((row['Timestamp'], "Overcurrent", "Current", f"{row['Current(A)']}A"))
            if counters['under_frequency'] == self.duration_threshold:
                faults.append((row['Timestamp'], "Under-frequency", "Frequency", f"{row['Frequency(Hz)']}Hz"))
            if counters['over_frequency'] == self.duration_threshold:
                faults.append((row['Timestamp'], "Over-frequency", "Frequency", f"{row['Frequency(Hz)']}Hz"))
            if counters['low_power_factor'] == self.duration_threshold:
                faults.append((row['Timestamp'], "Low Power Factor", "Power Factor", f"{row['PowerFactor']}"))

        return faults