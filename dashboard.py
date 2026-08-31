import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime
from detector import EnhancedFaultDetector
from visualizer import EnhancedDataVisualizer

st.set_page_config(
    page_title="Smart Grid Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .critical-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-alert {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-alert {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_components():
    return EnhancedFaultDetector(config_path="config.json"), EnhancedDataVisualizer()


detector, visualizer = initialize_components()

with open("config.json", "r") as f:
    config = json.load(f)

REFRESH_INTERVAL = 2
GRID_SECTION     = "Substation_A"
MAX_FAULTS       = 20

st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "Live Monitoring",
    "Analytics",
    "Historical Data",
    "System Settings",
])

# -----------------------------------------------------------------------
# Live Monitoring
# -----------------------------------------------------------------------
if page == "Live Monitoring":
    st.title("Smart Grid Live Monitoring")
    placeholder = st.empty()

    while True:
        try:
            data = pd.read_csv("live_data.csv")
            data.drop_duplicates(subset=["Timestamp"], keep="last", inplace=True)

            faults      = detector.detect_faults(data, GRID_SECTION)
            predictions = detector.predict_potential_faults(data)
            stats       = detector.get_fault_statistics()
            recent      = detector.get_recent_faults(MAX_FAULTS)

            with placeholder.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"### {datetime.now().strftime('%H:%M:%S')} — {GRID_SECTION}")
                with col2:
                    if stats["critical_last_hour"] > 0:
                        st.error(f"{stats['critical_last_hour']} Critical alerts (last hour)")
                with col3:
                    st.info(f"Refresh: {REFRESH_INTERVAL}s")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Readings", len(data))
                with m2:
                    active = len([f for f in faults if len(f) >= 5 and f[4] in ("critical", "warning")])
                    st.metric("Active Faults", active, delta=f"{active}", delta_color="inverse")
                with m3:
                    st.metric("Today's Alerts", stats["today_total"])
                with m4:
                    if data["Voltage(V)"].iloc[-1] > 200:
                        st.metric("Grid Health", "Good", delta="Stable")
                    else:
                        st.metric("Grid Health", "Warning", delta="Unstable", delta_color="inverse")

                viz_col, data_col = st.columns([2, 1])
                with viz_col:
                    st.markdown("#### Real-time Parameters")
                    fig = visualizer.plot_data(data, faults, detector.thresholds, predictions)
                    st.pyplot(fig)

                with data_col:
                    st.markdown("#### Recent Alerts")
                    if recent:
                        for fault in recent[:10]:
                            sev = fault.get("severity", "info")
                            st.markdown(f"""
                            <div class="{sev}-alert">
                                <strong>{fault['fault_type']}</strong><br>
                                <small>Time: {fault['timestamp']}</small><br>
                                <small>{fault['parameter']}: {fault['value']}</small><br>
                                <small>Section: {fault['grid_section']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("No recent alerts")

                if predictions:
                    st.markdown("#### Predictive Warnings")
                    pred_cols = st.columns(len(predictions))
                    for i, pred in enumerate(predictions):
                        with pred_cols[i]:
                            level = "HIGH" if pred["confidence"] > 70 else "MEDIUM" if pred["confidence"] > 40 else "LOW"
                            st.markdown(
                                f"**[{level}] {pred['type']}**  \n"
                                f"Confidence: {pred['confidence']:.1f}%  \n"
                                f"ETA: {pred['estimated_time']}"
                            )

                st.markdown("#### System Status")
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.success("Data Stream: Active")
                with s2:
                    if stats["critical_last_hour"] == 0:
                        st.success("Alert System: Normal")
                    else:
                        st.error("Alert System: Active Alerts")
                with s3:
                    st.info(f"Monitoring: {GRID_SECTION}")

        except FileNotFoundError:
            with placeholder.container():
                st.warning("Waiting for data stream. Run producer.py in a separate terminal.")
        except pd.errors.EmptyDataError:
            with placeholder.container():
                st.warning("Data file is empty, waiting for data...")
        except Exception as e:
            with placeholder.container():
                st.error(f"Error: {e}")

        time.sleep(REFRESH_INTERVAL)

# -----------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------
elif page == "Analytics":
    st.title("Grid Analytics")
    try:
        recent = detector.get_recent_faults(100)
        if recent:
            st.markdown("#### Severity Distribution")
            pie_fig = visualizer.plot_severity_distribution(recent)
            if pie_fig:
                st.pyplot(pie_fig)

            st.markdown("#### Trend Analysis")
            if st.button("Generate Trend Analysis"):
                try:
                    data = pd.read_csv("live_data.csv")
                    trend_fig = visualizer.plot_trend_analysis(data)
                    if trend_fig:
                        st.pyplot(trend_fig)
                except Exception:
                    st.warning("Unable to generate trend analysis — insufficient data.")
        else:
            st.info("No historical data available.")
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# -----------------------------------------------------------------------
# Historical Data
# -----------------------------------------------------------------------
elif page == "Historical Data":
    st.title("Historical Data")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Cleanup Old Data"):
            deleted = detector.cleanup_old_data()
            st.success(f"Cleaned up {deleted} old records.")
    with c2:
        st.number_input("Retention Period (days)", 1, 365,
                        config["data_settings"]["retention_days"])
    with c3:
        if st.button("Export Data"):
            recent = detector.get_recent_faults(1000)
            if recent:
                csv = pd.DataFrame(recent).to_csv(index=False)
                st.download_button("Download CSV", csv, "fault_history.csv", "text/csv")

    st.markdown("#### Recent Fault Records")
    recent = detector.get_recent_faults(50)
    if recent:
        st.dataframe(pd.DataFrame(recent), use_container_width=True)
    else:
        st.info("No historical data available.")

# -----------------------------------------------------------------------
# System Settings
# -----------------------------------------------------------------------
elif page == "System Settings":
    st.title("System Configuration")

    st.markdown("#### Fault Thresholds")
    threshold_rows = [
        {
            "Fault Type": k.replace("_", " ").title(),
            "Threshold":  str(v),
            "Severity":   v.get("severity", "info"),
        }
        for k, v in config["fault_thresholds"].items()
    ]
    st.dataframe(pd.DataFrame(threshold_rows), use_container_width=True)

    st.markdown("#### Alert Configuration")
    a1, a2 = st.columns(2)
    with a1:
        st.json(config["alert_settings"]["email_alerts"])
    with a2:
        st.json(config["alert_settings"]["sms_alerts"])

    st.markdown("#### System Information")
    for k, v in {
        "Configuration File": "config.json",
        "Database":           "alerts.db",
        "Data Retention":     f"{config['data_settings']['retention_days']} days",
        "Refresh Interval":   f"{REFRESH_INTERVAL} seconds",
    }.items():
        st.text(f"{k}: {v}")

    if st.button("Test Alert System"):
        test = {
            "timestamp":    datetime.now().isoformat(),
            "fault_type":   "Test Alert",
            "parameter":    "System Test",
            "value":        "Test Value",
            "grid_section": GRID_SECTION,
        }
        try:
            result = detector.alert_manager.process_fault(test)
            st.success("Test alert processed.")
            st.json(result)
        except Exception as e:
            st.error(f"Test alert failed: {e}")

st.markdown("---")
st.markdown("Smart Grid Monitoring System | Real-time fault detection with predictive analytics")
