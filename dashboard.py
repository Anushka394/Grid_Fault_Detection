import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime
from detector import EnhancedFaultDetector
from visualizer import EnhancedDataVisualizer

st.set_page_config(
    page_title="Smart Grid Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global ── */
    [data-testid="stAppViewContainer"] {
        background-color: #0f1117;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f45;
    }
    [data-testid="stSidebar"] * { color: #c9d1e0 !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #1c2133;
        border: 1px solid #2a2f45;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"]  { color: #8b9ab0 !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"]  { color: #e8eaf6 !important; font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricDelta"]  { font-size: 0.75rem; }

    /* ── Alert cards ── */
    .alert-card {
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid;
    }
    .alert-card.critical {
        background: #2a1a1f;
        border-color: #ef5350;
        color: #ffcdd2;
    }
    .alert-card.warning {
        background: #2a2010;
        border-color: #ffa726;
        color: #ffe0b2;
    }
    .alert-card.info {
        background: #0d1f2d;
        border-color: #29b6f6;
        color: #b3e5fc;
    }
    .alert-card strong { font-size: 0.9rem; display: block; margin-bottom: 3px; }
    .alert-card span   { font-size: 0.75rem; opacity: 0.85; display: block; line-height: 1.5; }

    /* ── Prediction badge ── */
    .pred-card {
        background: #1a1f35;
        border: 1px solid #2e3a5c;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        text-align: center;
    }
    .pred-card .pred-type  { font-size: 0.82rem; color: #90caf9; font-weight: 600; margin-bottom: 6px; }
    .pred-card .pred-conf  { font-size: 1.6rem; font-weight: 800; margin-bottom: 4px; }
    .pred-card .pred-conf.high   { color: #ef5350; }
    .pred-card .pred-conf.medium { color: #ffa726; }
    .pred-card .pred-conf.low    { color: #66bb6a; }
    .pred-card .pred-eta   { font-size: 0.75rem; color: #8b9ab0; }
    .pred-card .pred-badge {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    .pred-card .pred-badge.high   { background:#ef535022; color:#ef5350; border:1px solid #ef535055; }
    .pred-card .pred-badge.medium { background:#ffa72622; color:#ffa726; border:1px solid #ffa72655; }
    .pred-card .pred-badge.low    { background:#66bb6a22; color:#66bb6a; border:1px solid #66bb6a55; }

    /* ── Section headers ── */
    .section-header {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #5c6a82;
        margin-bottom: 0.6rem;
        margin-top: 0.2rem;
    }

    /* ── Status pill ── */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .status-pill.green  { background:#1b3a2a; color:#66bb6a; border:1px solid #66bb6a55; }
    .status-pill.red    { background:#3a1a1f; color:#ef5350; border:1px solid #ef535055; }
    .status-pill.blue   { background:#0d1f35; color:#29b6f6; border:1px solid #29b6f655; }

    /* ── Divider ── */
    hr { border-color: #2a2f45 !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* ── Hide streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* ── Sidebar nav radio ── */
    [data-testid="stRadio"] label {
        padding: 6px 10px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.88rem;
    }
    [data-testid="stRadio"] label:hover { background: #1f2640; }
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

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Smart Grid")
    st.markdown("<div style='font-size:0.75rem;color:#5c6a82;margin-bottom:1.5rem;'>Real-time Fault Monitor</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "Live Monitoring",
        "Analytics",
        "Historical Data",
        "System Settings",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"<div style='font-size:0.72rem;color:#5c6a82;'>Grid: {GRID_SECTION}<br>Refresh: every {REFRESH_INTERVAL}s</div>", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────
def alert_card(fault):
    sev = fault.get("severity", "info")
    icon = "🔴" if sev == "critical" else "🟠" if sev == "warning" else "🔵"
    return f"""
    <div class="alert-card {sev}">
        <strong>{icon} {fault['fault_type']}</strong>
        <span>T: {fault['timestamp']} &nbsp;|&nbsp; {fault['parameter']}: {fault['value']}</span>
    </div>"""


def pred_card(pred):
    c = pred["confidence"]
    level = "high" if c > 70 else "medium" if c > 40 else "low"
    return f"""
    <div class="pred-card">
        <div class="pred-badge {level}">{level.upper()}</div>
        <div class="pred-type">{pred['type']}</div>
        <div class="pred-conf {level}">{c:.0f}%</div>
        <div class="pred-eta">ETA: {pred['estimated_time']}</div>
    </div>"""


# ── LIVE MONITORING ───────────────────────────────────────────────────────
if page == "Live Monitoring":

    placeholder = st.empty()

    while True:
        try:
            data = pd.read_csv("live_data.csv")
            data.drop_duplicates(subset=["Timestamp"], keep="last", inplace=True)

            faults      = detector.detect_faults(data, GRID_SECTION)
            predictions = detector.predict_potential_faults(data)
            stats       = detector.get_fault_statistics()
            recent      = detector.get_recent_faults(MAX_FAULTS)

            last = data.iloc[-1]
            active_faults = len([f for f in faults if len(f) >= 5 and f[4] in ("critical", "warning")])
            grid_ok = float(last["Voltage(V)"]) > 200 and float(last["Current(A)"]) < 15

            with placeholder.container():

                # ── Top bar ──────────────────────────────────────────────
                tb_left, tb_right = st.columns([3, 1])
                with tb_left:
                    st.markdown(f"## Smart Grid Monitor")
                    st.markdown(f"<div style='font-size:0.8rem;color:#5c6a82;margin-top:-12px;'>Last updated: {datetime.now().strftime('%d %b %Y  %H:%M:%S')}</div>", unsafe_allow_html=True)
                with tb_right:
                    health_html = (
                        '<span class="status-pill green">GRID HEALTHY</span>'
                        if grid_ok else
                        '<span class="status-pill red">GRID WARNING</span>'
                    )
                    st.markdown(f"<div style='text-align:right;padding-top:14px;'>{health_html}</div>", unsafe_allow_html=True)

                st.markdown("---")

                # ── KPI row ───────────────────────────────────────────────
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                with k1:
                    st.metric("Voltage", f"{float(last['Voltage(V)']):.1f} V")
                with k2:
                    st.metric("Current", f"{float(last['Current(A)']):.1f} A")
                with k3:
                    st.metric("Frequency", f"{float(last['Frequency(Hz)']):.2f} Hz")
                with k4:
                    st.metric("Power Factor", f"{float(last['PowerFactor']):.3f}")
                with k5:
                    st.metric("Active Faults", active_faults)
                with k6:
                    st.metric("Today's Alerts", stats["today_total"])

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                # ── Charts + Alerts ───────────────────────────────────────
                chart_col, alert_col = st.columns([3, 1])

                with chart_col:
                    st.markdown('<div class="section-header">Parameter Trends</div>', unsafe_allow_html=True)
                    fig = visualizer.plot_data(data, faults, detector.thresholds, predictions)
                    st.pyplot(fig, use_container_width=True)

                with alert_col:
                    st.markdown('<div class="section-header">Recent Alerts</div>', unsafe_allow_html=True)
                    if recent:
                        cards_html = "".join(alert_card(f) for f in recent[:8])
                        st.markdown(cards_html, unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-card info"><strong>All Clear</strong><span>No alerts recorded</span></div>', unsafe_allow_html=True)

                # ── Predictions ───────────────────────────────────────────
                if predictions:
                    st.markdown("---")
                    st.markdown('<div class="section-header">Predictive Warnings</div>', unsafe_allow_html=True)
                    pcols = st.columns(max(len(predictions), 1))
                    for i, pred in enumerate(predictions):
                        with pcols[i]:
                            st.markdown(pred_card(pred), unsafe_allow_html=True)

                # ── Status bar ────────────────────────────────────────────
                st.markdown("---")
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.markdown('<span class="status-pill green">Data Stream Active</span>', unsafe_allow_html=True)
                with s2:
                    pill = '<span class="status-pill red">Critical Alerts</span>' if stats["critical_last_hour"] > 0 else '<span class="status-pill green">Alert System Normal</span>'
                    st.markdown(pill, unsafe_allow_html=True)
                with s3:
                    st.markdown(f'<span class="status-pill blue">Monitoring: {GRID_SECTION}</span>', unsafe_allow_html=True)
                with s4:
                    model_status = "Model Active" if predictions is not None else "Fallback Mode"
                    st.markdown(f'<span class="status-pill blue">{model_status}</span>', unsafe_allow_html=True)

        except FileNotFoundError:
            with placeholder.container():
                st.markdown("---")
                st.warning("Waiting for data stream — run `py producer.py` in a separate terminal.")
        except pd.errors.EmptyDataError:
            with placeholder.container():
                st.warning("Data file is empty, waiting...")
        except Exception as e:
            with placeholder.container():
                st.error(f"Error: {e}")

        time.sleep(REFRESH_INTERVAL)


# ── ANALYTICS ─────────────────────────────────────────────────────────────
elif page == "Analytics":
    st.markdown("## Grid Analytics")
    st.markdown("---")
    try:
        recent = detector.get_recent_faults(100)
        if recent:
            df_recent = pd.DataFrame(recent)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Faults Logged", len(df_recent))
            with col2:
                crit = len(df_recent[df_recent["severity"] == "critical"]) if "severity" in df_recent else 0
                st.metric("Critical", crit)
            with col3:
                warn = len(df_recent[df_recent["severity"] == "warning"]) if "severity" in df_recent else 0
                st.metric("Warnings", warn)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-header">Severity Distribution</div>', unsafe_allow_html=True)
                pie_fig = visualizer.plot_severity_distribution(recent)
                if pie_fig:
                    st.pyplot(pie_fig, use_container_width=True)
            with c2:
                st.markdown('<div class="section-header">Fault Type Breakdown</div>', unsafe_allow_html=True)
                if "fault_type" in df_recent.columns:
                    ft_counts = df_recent["fault_type"].value_counts().reset_index()
                    ft_counts.columns = ["Fault Type", "Count"]
                    st.dataframe(ft_counts, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown('<div class="section-header">Trend Analysis</div>', unsafe_allow_html=True)
            if st.button("Generate Trend Analysis"):
                try:
                    data = pd.read_csv("live_data.csv")
                    trend_fig = visualizer.plot_trend_analysis(data)
                    if trend_fig:
                        st.pyplot(trend_fig, use_container_width=True)
                except Exception:
                    st.warning("Unable to generate — insufficient data.")
        else:
            st.info("No historical data available yet.")
    except Exception as e:
        st.error(f"Error: {e}")


# ── HISTORICAL DATA ────────────────────────────────────────────────────────
elif page == "Historical Data":
    st.markdown("## Historical Fault Records")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Cleanup Old Data"):
            deleted = detector.cleanup_old_data()
            st.success(f"Removed {deleted} old records.")
    with c2:
        st.number_input("Retention Period (days)", 1, 365,
                        config["data_settings"]["retention_days"])
    with c3:
        if st.button("Export to CSV"):
            records = detector.get_recent_faults(1000)
            if records:
                csv = pd.DataFrame(records).to_csv(index=False)
                st.download_button("Download", csv, "fault_history.csv", "text/csv")

    st.markdown("---")
    recent = detector.get_recent_faults(50)
    if recent:
        df = pd.DataFrame(recent)

        # colour severity column
        def colour_severity(val):
            colours = {
                "critical": "background-color:#2a1a1f;color:#ef5350",
                "warning":  "background-color:#2a2010;color:#ffa726",
                "info":     "background-color:#0d1f2d;color:#29b6f6",
            }
            return colours.get(val, "")

        styled = df.style.map(colour_severity, subset=["severity"])
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("No records found.")


# ── SYSTEM SETTINGS ────────────────────────────────────────────────────────
elif page == "System Settings":
    st.markdown("## System Configuration")
    st.markdown("---")

    st.markdown('<div class="section-header">Fault Detection Thresholds</div>', unsafe_allow_html=True)
    threshold_rows = [
        {
            "Fault Type": k.replace("_", " ").title(),
            "Threshold":  str({sk: sv for sk, sv in v.items() if sk != "severity"}),
            "Severity":   v.get("severity", "info"),
        }
        for k, v in config["fault_thresholds"].items()
    ]
    st.dataframe(pd.DataFrame(threshold_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Alert Configuration</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Email Alerts**")
        st.json(config["alert_settings"]["email_alerts"])
    with a2:
        st.markdown("**SMS Alerts**")
        st.json(config["alert_settings"]["sms_alerts"])

    st.markdown("---")
    st.markdown('<div class="section-header">System Information</div>', unsafe_allow_html=True)
    info_data = {
        "Configuration File": "config.json",
        "Database":           "alerts.db",
        "Data Retention":     f"{config['data_settings']['retention_days']} days",
        "Refresh Interval":   f"{REFRESH_INTERVAL} seconds",
        "Grid Section":       GRID_SECTION,
    }
    for k, v in info_data.items():
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"<span style='color:#5c6a82;font-size:0.85rem;'>{k}</span>", unsafe_allow_html=True)
        with r2:
            st.markdown(f"<span style='color:#e0e0e0;font-size:0.85rem;'>{v}</span>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Run Test Alert"):
        test = {
            "timestamp":    datetime.now().isoformat(),
            "fault_type":   "Test Alert",
            "parameter":    "System Test",
            "value":        "Test Value",
            "grid_section": GRID_SECTION,
        }
        try:
            result = detector.alert_manager.process_fault(test)
            st.success("Test alert processed and logged.")
            st.json(result)
        except Exception as e:
            st.error(f"Test failed: {e}")
