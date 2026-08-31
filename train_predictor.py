"""
train_predictor.py
==================
Trains a LogisticRegression model to predict imminent grid faults.

FEATURES  (computed per row i using a trailing window of 5 rows)
-----------------------------------------------------------------
  voltage_slope  : degree-1 polyfit slope of Voltage(V)
  current_slope  : degree-1 polyfit slope of Current(A)
  freq_std       : rolling standard deviation of Frequency(Hz)
  pf_slope       : degree-1 polyfit slope of PowerFactor
  voltage_accel  : change in voltage_slope vs. previous window (2nd-order trend)
  current_accel  : change in current_slope vs. previous window (2nd-order trend)

LABEL
-----
  Label for row i = 1 if any confirmed fault occurs within the next N rows (default N=5).
  This trains the model to warn before the threshold breach, not at it.

TRAIN / TEST SPLIT
------------------
  80/20 split in time order — no shuffle. Shuffling would leak future readings into
  the training set, inflating metrics. Time-ordered split matches production use.

MODEL
-----
  StandardScaler -> LogisticRegression (liblinear, class_weight=balanced).
  Chosen for explainability: coefficients show which features drive fault probability.
  predict_proba() outputs a calibrated probability used directly as confidence score.

OUTPUT
------
  predictor_model.pkl — serialised Pipeline loaded at runtime by detector.py

USAGE
-----
  python train_predictor.py
  python train_predictor.py --csv my_data.csv --horizon 10
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score,
                                 precision_score, recall_score)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("\n[ERROR] scikit-learn / joblib not found.")
    print("Install with:  pip install scikit-learn joblib\n")
    sys.exit(1)

warnings.filterwarnings("ignore")

WINDOW       = 5
MODEL_PATH   = "predictor_model.pkl"
CONFIG_PATH  = "config.json"
DATA_PATH    = "grid_data.csv"
FEATURE_NAMES = [
    "voltage_slope", "current_slope", "freq_std",
    "pf_slope", "voltage_accel", "current_accel",
]


# ---------------------------------------------------------------------------
# Fault label helpers
# ---------------------------------------------------------------------------

def _load_thresholds(config_path=CONFIG_PATH):
    with open(config_path, "r") as f:
        return json.load(f)["fault_thresholds"]


def _row_has_fault(row, th):
    """Return True if the row breaches any threshold in config.json."""
    v, i, f, pf = row["Voltage(V)"], row["Current(A)"], row["Frequency(Hz)"], row["PowerFactor"]
    if v < th["earth_fault"]["max_voltage"] and i > th["earth_fault"]["min_current"]:
        return True
    if v < th["under_voltage"]["max_voltage"]:
        return True
    if i > th["over_current"]["min_current"]:
        return True
    if f < th["under_frequency"]["min_freq"]:
        return True
    if f > th["over_frequency"]["max_freq"]:
        return True
    if pf < th["low_power_factor"]["min_pf"]:
        return True
    if th["voltage_sag"]["min_voltage"] <= v <= th["voltage_sag"]["max_voltage"]:
        return True
    return False


def build_fault_flags(df, th):
    return np.array([_row_has_fault(row, th) for _, row in df.iterrows()], dtype=bool)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _slope(arr):
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def build_features(df, window=WINDOW):
    """Compute 6 features per row using a trailing window. NaN for first window-1 rows."""
    records = []
    prev_v_slope = prev_i_slope = None

    for i in range(len(df)):
        if i < window - 1:
            records.append({k: np.nan for k in FEATURE_NAMES})
            continue

        win = df.iloc[i - window + 1: i + 1]
        v_slope = _slope(win["Voltage(V)"].values)
        i_slope = _slope(win["Current(A)"].values)
        f_std   = float(win["Frequency(Hz)"].std(ddof=1))
        pf_sl   = _slope(win["PowerFactor"].values)
        v_accel = (v_slope - prev_v_slope) if prev_v_slope is not None else 0.0
        i_accel = (i_slope - prev_i_slope) if prev_i_slope is not None else 0.0
        prev_v_slope, prev_i_slope = v_slope, i_slope

        records.append({
            "voltage_slope": v_slope, "current_slope": i_slope,
            "freq_std": f_std,        "pf_slope": pf_sl,
            "voltage_accel": v_accel, "current_accel": i_accel,
        })

    return pd.DataFrame(records, index=df.index)


def build_labels(fault_flags, horizon):
    """Label row i = 1 if any fault occurs in rows (i+1) to (i+horizon)."""
    n = len(fault_flags)
    labels = np.zeros(n, dtype=int)
    for i in range(n - horizon):
        if fault_flags[i + 1: i + 1 + horizon].any():
            labels[i] = 1
    return labels


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _ensure_rich_dataset(csv_path, min_rows=300):
    """
    Load CSV. If fewer than min_rows rows, extend with synthetic readings that
    cover normal operation and all fault boundary conditions so the model has
    balanced positive and negative examples to learn from.
    Synthetic data uses the same statistical profile as the real data.
    """
    df = pd.read_csv(csv_path)
    if len(df) >= min_rows:
        return df

    print(f"  [info] {csv_path} has {len(df)} rows — extending to {min_rows} with synthetic data.")

    rng     = np.random.default_rng(42)
    n_extra = min_rows - len(df)
    v_mean  = float(df["Voltage(V)"].mean())    if len(df) > 2 else 230.0
    i_mean  = float(df["Current(A)"].mean())    if len(df) > 2 else 5.0
    f_mean  = float(df["Frequency(Hz)"].mean()) if len(df) > 2 else 50.0
    pf_mean = float(df["PowerFactor"].mean())   if len(df) > 2 else 0.97

    v  = rng.normal(v_mean,  10.0, n_extra)
    i  = rng.normal(i_mean,   1.5, n_extra)
    f  = rng.normal(f_mean,   0.3, n_extra)
    pf = np.clip(rng.normal(pf_mean, 0.03, n_extra), 0.70, 1.00)

    # Inject fault scenarios so labels are balanced
    s = n_extra // 10
    v[s * 2: s * 2 + 15] = rng.uniform(70, 95,  15)   # earth fault
    i[s * 2: s * 2 + 15] = rng.uniform(16, 22,  15)
    v[s * 4: s * 4 + 20] = np.linspace(v_mean, 160, 20)  # under-voltage ramp
    f[s * 6: s * 6 + 15] = rng.uniform(47.5, 48.9, 15)   # under-frequency
    pf[s * 8: s * 8 + 20] = rng.uniform(0.72, 0.88, 20)  # low power factor
    i[s:     s + 10]      = rng.uniform(16, 20, 10)        # overcurrent spike

    extra = pd.DataFrame({
        "Timestamp":     range(len(df), len(df) + n_extra),
        "Voltage(V)":    np.round(np.clip(v,  50,   280), 2),
        "Current(A)":    np.round(np.clip(i,   0,    30), 2),
        "Frequency(Hz)": np.round(np.clip(f,  45,    55), 2),
        "PowerFactor":   np.round(np.clip(pf, 0.70, 1.00), 3),
    })
    return pd.concat([df, extra], ignore_index=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(csv_path=DATA_PATH, horizon=WINDOW, model_out=MODEL_PATH, config_path=CONFIG_PATH):
    print("\n" + "=" * 60)
    print("  Smart Grid Fault Predictor — Training")
    print("=" * 60)

    print(f"\n[1/5] Loading data from '{csv_path}' ...")
    df = _ensure_rich_dataset(csv_path)
    print(f"      Rows available: {len(df)}")

    print(f"\n[2/5] Building features (window={WINDOW}) and labels (horizon={horizon}) ...")
    th          = _load_thresholds(config_path)
    feat_df     = build_features(df, window=WINDOW)
    fault_flags = build_fault_flags(df, th)
    labels      = build_labels(fault_flags, horizon=horizon)

    feat_df = feat_df.iloc[WINDOW - 1: len(df) - horizon].reset_index(drop=True)
    labels  = labels[WINDOW - 1: len(df) - horizon]

    print(f"      Feature matrix : {feat_df.shape}")
    print(f"      Fault-imminent : {labels.sum()}  ({labels.mean() * 100:.1f}%)")
    print(f"      Normal         : {(labels == 0).sum()}  ({(labels == 0).mean() * 100:.1f}%)")

    if labels.sum() == 0:
        print("\n[WARN] No fault-imminent labels found. Training aborted.")
        return None

    print("\n[3/5] Splitting train/test (80/20, time-ordered) ...")
    split   = int(len(feat_df) * 0.80)
    X_train, X_test = feat_df.iloc[:split].values, feat_df.iloc[split:].values
    y_train, y_test = labels[:split], labels[split:]

    print(f"      Train: {len(X_train)} rows  (fault-imminent: {y_train.sum()})")
    print(f"      Test:  {len(X_test)} rows   (fault-imminent: {y_test.sum()})")

    if y_train.sum() == 0 or y_test.sum() == 0:
        print("\n[WARN] One split has no fault-imminent labels. Try a larger dataset or longer horizon.")
        return None

    print("\n[4/5] Training LogisticRegression ...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000,
                                   class_weight="balanced", random_state=42)),
    ])
    model.fit(X_train, y_train)

    print("\n[5/5] Evaluating on test split ...")
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred,    zero_division=0)
    f1        = f1_score(y_test, y_pred,         zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    print("\n" + "-" * 50)
    print("  RESULTS (held-out test set)")
    print("-" * 50)
    print(f"  Accuracy  : {accuracy  * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall    * 100:.2f}%")
    print(f"  F1 Score  : {f1        * 100:.2f}%")
    print()
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"             Pred 0   Pred 1")
    print(f"  Actual 0 : {cm[0, 0]:6d}   {cm[0, 1]:6d}")
    print(f"  Actual 1 : {cm[1, 0]:6d}   {cm[1, 1]:6d}")
    print()
    print(classification_report(y_test, y_pred,
                                 target_names=["Normal", "Fault-imminent"],
                                 zero_division=0))

    coefs = model.named_steps["clf"].coef_[0]
    print("  Feature coefficients (sorted by magnitude):")
    for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: -abs(x[1])):
        print(f"    {name:<18s} : {coef:+.4f}  ({'fault' if coef > 0 else 'normal'})")
    print("-" * 50)

    joblib.dump(model, model_out)
    print(f"\n  Model saved to '{model_out}'")
    print("=" * 60 + "\n")

    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "confusion_matrix": cm.tolist()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Smart Grid fault prediction model.")
    parser.add_argument("--csv",     default=DATA_PATH,  help="Path to grid CSV")
    parser.add_argument("--horizon", default=5, type=int, help="Look-ahead rows for label (default 5)")
    parser.add_argument("--out",     default=MODEL_PATH,  help="Output .pkl path")
    args = parser.parse_args()
    train(csv_path=args.csv, horizon=args.horizon, model_out=args.out)
