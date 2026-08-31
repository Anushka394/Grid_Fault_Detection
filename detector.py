import logging
import os
import json
import numpy as np
import pandas as pd
from alert_manager import AlertManager

logger = logging.getLogger(__name__)

# Trained model cache — loaded once at startup, never reloaded per call
_PREDICTOR_MODEL = None
_PREDICTOR_LOAD_ATTEMPTED = False
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "predictor_model.pkl")

_PREDICT_FEATURES = [
    "voltage_slope", "current_slope", "freq_std",
    "pf_slope", "voltage_accel", "current_accel",
]

_WINDOW = 5  # must match train_predictor.py


def _load_predictor_model():
    """Load predictor_model.pkl once and cache it. Returns None if missing or unloadable."""
    global _PREDICTOR_MODEL, _PREDICTOR_LOAD_ATTEMPTED
    if _PREDICTOR_LOAD_ATTEMPTED:
        return _PREDICTOR_MODEL
    _PREDICTOR_LOAD_ATTEMPTED = True

    if not os.path.exists(_MODEL_PATH):
        logger.warning(
            "[PREDICTION] predictor_model.pkl not found. Running in FALLBACK mode — "
            "confidence scores are heuristic estimates. Run python train_predictor.py to train the model."
        )
        return None

    try:
        import joblib
        _PREDICTOR_MODEL = joblib.load(_MODEL_PATH)
        logger.info("[PREDICTION] Loaded predictor_model.pkl from '%s'.", _MODEL_PATH)
    except Exception as exc:
        logger.error("[PREDICTION] Failed to load predictor_model.pkl: %s. Using fallback mode.", exc)
        _PREDICTOR_MODEL = None

    return _PREDICTOR_MODEL


def _compute_features(recent_data, prev_v_slope, prev_i_slope):
    """Compute the 6 prediction features from a 5-row window."""
    x = np.arange(_WINDOW, dtype=float)
    v_slope = float(np.polyfit(x, recent_data["Voltage(V)"].values,  1)[0])
    i_slope = float(np.polyfit(x, recent_data["Current(A)"].values,  1)[0])
    f_std   = float(recent_data["Frequency(Hz)"].std(ddof=1))
    pf_sl   = float(np.polyfit(x, recent_data["PowerFactor"].values, 1)[0])
    v_accel = (v_slope - prev_v_slope) if prev_v_slope is not None else 0.0
    i_accel = (i_slope - prev_i_slope) if prev_i_slope is not None else 0.0
    return (
        {"voltage_slope": v_slope, "current_slope": i_slope, "freq_std": f_std,
         "pf_slope": pf_sl, "voltage_accel": v_accel, "current_accel": i_accel},
        v_slope, i_slope,
    )


def _time_to_breach(slope_magnitude, per_step_seconds=2):
    """Estimate time to threshold breach from slope magnitude."""
    if slope_magnitude < 1e-6:
        return "5-10 minutes"
    steps = max(1, int(20 / slope_magnitude))
    lo = max(1, (steps * per_step_seconds) // 60)
    hi = lo + max(2, lo // 2)
    return f"{lo}-{hi} minutes"


class EnhancedFaultDetector:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.thresholds = self.config["fault_thresholds"]
        self.duration_threshold = self.config.get("fault_duration_threshold", 1)
        self.alert_manager = AlertManager(config_path)
        self.fault_counters = {}

    def detect_faults(self, data, grid_section="Substation_A"):
        """Threshold-based detection for all confirmed fault types."""
        faults = []

        if grid_section not in self.fault_counters:
            self.fault_counters[grid_section] = {
                "under_voltage": 0, "over_current": 0, "under_frequency": 0,
                "over_frequency": 0, "low_power_factor": 0, "voltage_sag": 0,
            }

        counters = self.fault_counters[grid_section]
        th = self.thresholds

        for _, row in data.iterrows():
            fault_found = False

            # Earth fault: simultaneous low voltage and high current
            if (row["Voltage(V)"] < th["earth_fault"]["max_voltage"] and
                    row["Current(A)"] > th["earth_fault"]["min_current"]):
                fault_data = {
                    "timestamp": row["Timestamp"],
                    "fault_type": "Earth Fault",
                    "parameter": "Voltage/Current",
                    "value": f"{row['Voltage(V)']}V / {row['Current(A)']}A",
                    "grid_section": grid_section,
                }
                processed = self.alert_manager.process_fault(fault_data)
                faults.append((
                    processed["timestamp"], processed["fault_type"],
                    processed["parameter"], processed["value"], processed["severity"],
                ))
                fault_found = True

            if fault_found:
                for key in counters:
                    counters[key] = 0
                continue

            fault_checks = [
                ("under_voltage",    row["Voltage(V)"]   < th["under_voltage"]["max_voltage"],
                 "Voltage",      f"{row['Voltage(V)']}V"),
                ("over_current",     row["Current(A)"]   > th["over_current"]["min_current"],
                 "Current",      f"{row['Current(A)']}A"),
                ("under_frequency",  row["Frequency(Hz)"] < th["under_frequency"]["min_freq"],
                 "Frequency",    f"{row['Frequency(Hz)']}Hz"),
                ("over_frequency",   row["Frequency(Hz)"] > th["over_frequency"]["max_freq"],
                 "Frequency",    f"{row['Frequency(Hz)']}Hz"),
                ("low_power_factor", row["PowerFactor"]  < th["low_power_factor"]["min_pf"],
                 "Power Factor", f"{row['PowerFactor']}"),
                ("voltage_sag",
                 th["voltage_sag"]["min_voltage"] <= row["Voltage(V)"] <= th["voltage_sag"]["max_voltage"],
                 "Voltage",      f"{row['Voltage(V)']}V"),
            ]

            for fault_type, condition, param, value in fault_checks:
                if condition:
                    counters[fault_type] += 1
                else:
                    counters[fault_type] = 0

                if counters[fault_type] == self.duration_threshold:
                    fault_data = {
                        "timestamp": row["Timestamp"],
                        "fault_type": self._display_name(fault_type),
                        "parameter": param,
                        "value": value,
                        "grid_section": grid_section,
                    }
                    processed = self.alert_manager.process_fault(fault_data)
                    faults.append((
                        processed["timestamp"], processed["fault_type"],
                        processed["parameter"], processed["value"], processed["severity"],
                    ))

        return faults

    def _display_name(self, fault_type):
        names = {
            "under_voltage":    "Under-voltage",
            "over_current":     "Overcurrent",
            "under_frequency":  "Under-frequency",
            "over_frequency":   "Over-frequency",
            "low_power_factor": "Low Power Factor",
            "voltage_sag":      "Voltage Sag",
        }
        return names.get(fault_type, fault_type.title())

    def detect_harmonics(self, data):
        """
        Detects power quality degradation using RMS-level proxies.

        True THD requires waveform sampling at high frequency followed by FFT.
        This dataset provides one RMS reading per timestamp, which is insufficient
        for a real THD calculation. Instead, three indicators are used:

        1. Current Coefficient of Variation (rolling std / mean x 100): non-linear
           loads that cause harmonics also produce irregular current fluctuations.
        2. V-I divergence: current fluctuating disproportionately relative to
           voltage, especially with low power factor, indicates reactive loading.
        3. Power factor penalty: low PF adds weight to the combined score.

        The result is labelled "Power Quality Degradation" to reflect what is
        actually measured, not a computed THD value.
        """
        faults = []
        if len(data) < 3:
            return faults

        max_thd = self.thresholds["harmonics"]["max_thd"]

        for i, (_, row) in enumerate(data.iterrows()):
            if i < 2:
                continue

            window = data.iloc[max(0, i - 2): i + 1]

            current_std  = float(window["Current(A)"].std(ddof=1))
            current_mean = float(window["Current(A)"].mean())
            current_cv   = (current_std / current_mean * 100) if current_mean > 0 else 0.0

            v_range      = float(window["Voltage(V)"].max() - window["Voltage(V)"].min())
            i_range      = float(window["Current(A)"].max() - window["Current(A)"].min())
            pf_mean      = float(window["PowerFactor"].mean())
            vi_divergence = max(0.0, i_range - v_range / 50.0)
            pf_penalty    = max(0.0, (1.0 - pf_mean) * 20)
            proxy_score   = (current_cv * 0.6) + (vi_divergence * 1.5) + pf_penalty

            if proxy_score > max_thd:
                fault_data = {
                    "timestamp":    row["Timestamp"],
                    "fault_type":   "Power Quality Degradation",
                    "parameter":    "Current CV / V-I Divergence",
                    "value":        (f"CV={current_cv:.1f}% "
                                     f"V-I div={vi_divergence:.2f} "
                                     f"PF={pf_mean:.3f} "
                                     f"[score={proxy_score:.1f}%]"),
                    "grid_section": "Substation_A",
                }
                processed = self.alert_manager.process_fault(fault_data)
                faults.append((
                    processed["timestamp"], processed["fault_type"],
                    processed["parameter"], processed["value"], processed["severity"],
                ))

        return faults

    def predict_potential_faults(self, data):
        """
        Returns a list of predictive warnings, each with keys:
            type, confidence (0-100), estimated_time, parameter

        If predictor_model.pkl is present, confidence = predict_proba() x 100.
        If missing, falls back to a z-score sigmoid heuristic and logs a WARNING.
        """
        predictions = []

        if len(data) < _WINDOW:
            return predictions

        features, _, _ = _compute_features(data.tail(_WINDOW), None, None)
        voltage_slope = features["voltage_slope"]
        current_slope = features["current_slope"]
        freq_std      = features["freq_std"]

        model = _load_predictor_model()

        if model is not None:
            feat_vec   = np.array([[features[k] for k in _PREDICT_FEATURES]])
            prob_fault = float(model.predict_proba(feat_vec)[0, 1])
            confidence = round(prob_fault * 100, 1)

            if confidence > 0:
                if voltage_slope < -2:
                    predictions.append({
                        "type": "Potential Under-voltage",
                        "confidence": confidence,
                        "estimated_time": _time_to_breach(abs(voltage_slope)),
                        "parameter": "Voltage",
                    })
                if current_slope > 1:
                    predictions.append({
                        "type": "Potential Overcurrent",
                        "confidence": confidence,
                        "estimated_time": _time_to_breach(current_slope),
                        "parameter": "Current",
                    })
                if freq_std > 0.3:
                    predictions.append({
                        "type": "Frequency Instability",
                        "confidence": confidence,
                        "estimated_time": _time_to_breach(freq_std),
                        "parameter": "Frequency",
                    })
                # Model flags high probability but no single feature exceeds its threshold
                if not predictions and prob_fault >= 0.50:
                    dominant = max(
                        [("voltage_slope", abs(voltage_slope) * 10),
                         ("current_slope", current_slope * 15),
                         ("freq_std",      freq_std * 100)],
                        key=lambda t: t[1],
                    )
                    param_map = {
                        "voltage_slope": ("Potential Under-voltage", "Voltage",    abs(voltage_slope)),
                        "current_slope": ("Potential Overcurrent",   "Current",    current_slope),
                        "freq_std":      ("Frequency Instability",   "Frequency",  freq_std),
                    }
                    ptype, pparam, mag = param_map[dominant[0]]
                    predictions.append({
                        "type": ptype,
                        "confidence": confidence,
                        "estimated_time": _time_to_breach(mag),
                        "parameter": pparam,
                    })

        else:
            # Fallback: z-score sigmoid heuristic
            # Maps each feature's deviation from a normal baseline through a sigmoid
            # to produce a 0-100 score. Baseline: voltage_slope ~ 0±1, current_slope ~ 0±0.5,
            # freq_std ~ 0.1±0.08.

            def _zscore_conf(value, mean, std, cap=90.0):
                z = (value - mean) / max(std, 1e-9)
                prob = 1.0 / (1.0 + np.exp(-z))
                return round(min(prob * 100, cap), 1)

            if voltage_slope < -1:
                conf = _zscore_conf(-voltage_slope, 0, 1.0)
                if conf >= 50:
                    predictions.append({
                        "type": "Potential Under-voltage",
                        "confidence": conf,
                        "estimated_time": _time_to_breach(abs(voltage_slope)),
                        "parameter": "Voltage",
                    })
            if current_slope > 0.5:
                conf = _zscore_conf(current_slope, 0, 0.5)
                if conf >= 50:
                    predictions.append({
                        "type": "Potential Overcurrent",
                        "confidence": conf,
                        "estimated_time": _time_to_breach(current_slope),
                        "parameter": "Current",
                    })
            if freq_std > 0.1:
                conf = _zscore_conf(freq_std, 0.1, 0.08)
                if conf >= 50:
                    predictions.append({
                        "type": "Frequency Instability",
                        "confidence": conf,
                        "estimated_time": _time_to_breach(freq_std),
                        "parameter": "Frequency",
                    })

        return predictions

    def get_fault_statistics(self):
        return self.alert_manager.get_alert_statistics()

    def get_recent_faults(self, limit=20):
        return self.alert_manager.get_recent_alerts(limit)

    def cleanup_old_data(self):
        return self.alert_manager.cleanup_old_alerts()
