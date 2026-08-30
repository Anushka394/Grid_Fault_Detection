import logging
import os
import pandas as pd
import json
import numpy as np
from datetime import datetime
from alert_manager import AlertManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model cache — loaded once per process, never reloaded on every call
# ---------------------------------------------------------------------------
_PREDICTOR_MODEL = None          # sklearn Pipeline once loaded
_PREDICTOR_LOAD_ATTEMPTED = False  # so we don't retry a missing file repeatedly
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "predictor_model.pkl")

_PREDICT_FEATURES = [
    "voltage_slope", "current_slope", "freq_std",
    "pf_slope", "voltage_accel", "current_accel",
]

_WINDOW = 5   # trailing window — must match train_predictor.py


def _load_predictor_model():
    """
    Load predictor_model.pkl once and cache it in the module-level variable.
    Returns the Pipeline on success, None if the file doesn't exist.
    Logs a clear WARNING in fallback mode so operators know it's uncalibrated.
    """
    global _PREDICTOR_MODEL, _PREDICTOR_LOAD_ATTEMPTED
    if _PREDICTOR_LOAD_ATTEMPTED:
        return _PREDICTOR_MODEL
    _PREDICTOR_LOAD_ATTEMPTED = True

    if not os.path.exists(_MODEL_PATH):
        logger.warning(
            "[PREDICTION] predictor_model.pkl not found at '%s'. "
            "Running in FALLBACK / UNCALIBRATED mode — confidence scores are "
            "heuristic estimates, not calibrated probabilities. "
            "Run  python train_predictor.py  to train and save the model.",
            _MODEL_PATH,
        )
        return None

    try:
        import joblib
        _PREDICTOR_MODEL = joblib.load(_MODEL_PATH)
        logger.info("[PREDICTION] Loaded predictor_model.pkl from '%s'.", _MODEL_PATH)
    except Exception as exc:
        logger.error(
            "[PREDICTION] Failed to load predictor_model.pkl ('%s'): %s. "
            "Falling back to heuristic mode.", _MODEL_PATH, exc
        )
        _PREDICTOR_MODEL = None

    return _PREDICTOR_MODEL


def _compute_prediction_features(recent_data: pd.DataFrame,
                                   prev_v_slope: float | None,
                                   prev_i_slope: float | None) -> dict:
    """
    Compute the six features used by the trained model from a 5-row window.
    Also returns updated slope values for acceleration on the next call.
    """
    x = np.arange(_WINDOW, dtype=float)

    v_slope = float(np.polyfit(x, recent_data["Voltage(V)"].values,  1)[0])
    i_slope = float(np.polyfit(x, recent_data["Current(A)"].values,  1)[0])
    f_std   = float(recent_data["Frequency(Hz)"].std(ddof=1))
    pf_sl   = float(np.polyfit(x, recent_data["PowerFactor"].values, 1)[0])

    v_accel = (v_slope - prev_v_slope) if prev_v_slope is not None else 0.0
    i_accel = (i_slope - prev_i_slope) if prev_i_slope is not None else 0.0

    return (
        {
            "voltage_slope":  v_slope,
            "current_slope":  i_slope,
            "freq_std":       f_std,
            "pf_slope":       pf_sl,
            "voltage_accel":  v_accel,
            "current_accel":  i_accel,
        },
        v_slope,
        i_slope,
    )

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
        """
        Predictive analysis for potential faults.

        Model path: uses a trained LogisticRegression Pipeline serialised to
        predictor_model.pkl (train with  python train_predictor.py).

        If the model file is missing the method falls back to the original
        heuristic formula and logs a WARNING so operators are never silently
        running uncalibrated scores.

        Return shape (unchanged): list of dicts with keys
            'type', 'confidence', 'estimated_time', 'parameter'
        where 'confidence' is 0-100.  In model mode it is predict_proba()
        scaled to 0-100; in fallback mode it is the original heuristic value.
        """
        predictions = []

        if len(data) < _WINDOW:
            return predictions

        recent_data = data.tail(_WINDOW)

        # ------------------------------------------------------------------
        # Always compute the base feature values — needed by both the model
        # path and the fallback heuristic path, and for parameter selection.
        # ------------------------------------------------------------------
        features, _, _ = _compute_prediction_features(recent_data, None, None)

        voltage_slope = features["voltage_slope"]
        current_slope = features["current_slope"]
        freq_std      = features["freq_std"]

        # ------------------------------------------------------------------
        # Attempt to use the trained model
        # ------------------------------------------------------------------
        model = _load_predictor_model()

        if model is not None:
            # Build feature vector in the exact column order the model was trained on
            feat_vec = np.array([[features[k] for k in _PREDICT_FEATURES]])
            prob_fault = float(model.predict_proba(feat_vec)[0, 1])  # P(fault-imminent)
            confidence = round(prob_fault * 100, 1)

            # Decide which parameters to flag based on feature magnitudes,
            # exactly mirroring the original parameter-selection logic —
            # only the confidence number and the gate come from the model.
            if confidence > 0:   # model says non-trivial probability
                if voltage_slope < -2:
                    predictions.append({
                        'type': 'Potential Under-voltage',
                        'confidence': confidence,
                        'estimated_time': '5-10 minutes',
                        'parameter': 'Voltage'
                    })
                if current_slope > 1:
                    predictions.append({
                        'type': 'Potential Overcurrent',
                        'confidence': confidence,
                        'estimated_time': '3-8 minutes',
                        'parameter': 'Current'
                    })
                if freq_std > 0.3:
                    predictions.append({
                        'type': 'Frequency Instability',
                        'confidence': confidence,
                        'estimated_time': '2-5 minutes',
                        'parameter': 'Frequency'
                    })
                # If the model flags high probability but no single feature
                # exceeds a specific threshold, surface a generic warning so
                # the prediction is never silently swallowed.
                if not predictions and prob_fault >= 0.50:
                    dominant = max(
                        [("voltage_slope", abs(voltage_slope) * 10),
                         ("current_slope", current_slope * 15),
                         ("freq_std",      freq_std * 100)],
                        key=lambda t: t[1]
                    )
                    param_map = {
                        "voltage_slope": ("Potential Under-voltage", "Voltage", "5-10 minutes"),
                        "current_slope": ("Potential Overcurrent",   "Current", "3-8 minutes"),
                        "freq_std":      ("Frequency Instability",   "Frequency", "2-5 minutes"),
                    }
                    ptype, pparam, ptime = param_map[dominant[0]]
                    predictions.append({
                        'type': ptype,
                        'confidence': confidence,
                        'estimated_time': ptime,
                        'parameter': pparam,
                    })

        else:
            # ------------------------------------------------------------------
            # FALLBACK: original heuristic formula (uncalibrated)
            # ------------------------------------------------------------------
            if voltage_slope < -2:
                predictions.append({
                    'type': 'Potential Under-voltage',
                    'confidence': min(abs(voltage_slope) * 10, 95),
                    'estimated_time': '5-10 minutes',
                    'parameter': 'Voltage'
                })
            if current_slope > 1:
                predictions.append({
                    'type': 'Potential Overcurrent',
                    'confidence': min(current_slope * 15, 90),
                    'estimated_time': '3-8 minutes',
                    'parameter': 'Current'
                })
            if freq_std > 0.3:
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