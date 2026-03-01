import streamlit as st
import pandas as pd
import time
import json
from detector import EnhancedFaultDetector
from visualizer import EnhancedDataVisualizer
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Smart Grid Monitoring", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
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

# Initialize components
@st.cache_resource
def initialize_components():
    detector = EnhancedFaultDetector(config_path='config.json')
    visualizer = EnhancedDataVisualizer()
    return detector, visualizer

detector, visualizer = initialize_components()

# Sidebar configuration
st.sidebar.title("System Configuration")

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Simple settings - minimal
refresh_interval = 2  # Fixed 2 seconds
selected_section = "Substation_A"  # Default section
show_predictions = True  # Always show
max_faults_display = 20  # Fixed 20

# Navigation
page = st.sidebar.radio("Navigation", [
    "Live Monitoring", 
    "Analytics", 
    "Historical Data",
    "System Settings"
])

# Main content based on selected page
if page == "Live Monitoring":
    st.title("Smart Grid Monitoring")
    st.markdown(f"**Current Grid Section:** {selected_section}")
    
    # Create placeholder for live updates
    placeholder = st.empty()
    
    # Auto-refresh loop
    while True:
        try:
            # Read live data
            data = pd.read_csv('live_data.csv')
            data.drop_duplicates(subset=['Timestamp'], keep='last', inplace=True)
            
            # Detect faults
            faults = detector.detect_faults(data, selected_section)
            
            # Get predictions if enabled
            predictions = detector.predict_potential_faults(data) if show_predictions else None
            
            # Get statistics
            stats = detector.get_fault_statistics()
            recent_faults = detector.get_recent_faults(max_faults_display)
            
            with placeholder.container():
                # Header with timestamp
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"### Live Status - {datetime.now().strftime('%H:%M:%S')}")
                with col2:
                    if stats['critical_last_hour'] > 0:
                        st.error(f"ALERT: {stats['critical_last_hour']} Critical Alerts (Last Hour)")
                with col3:
                    st.info(f"Auto-refresh: {refresh_interval}s")
                
                # Key metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Total Readings", len(data))
                
                with metric_col2:
                    current_faults = len([f for f in faults if len(f) >= 5 and f[4] in ['critical', 'warning']])
                    st.metric("Active Faults", current_faults, 
                             delta=f"{current_faults} faults", 
                             delta_color="inverse")
                
                with metric_col3:
                    st.metric("Today's Alerts", stats['today_total'])
                
                with metric_col4:
                    if data['Voltage(V)'].iloc[-1] > 200:
                        st.metric("Grid Health", "Good", delta="Stable")
                    else:
                        st.metric("Grid Health", "Warning", delta="Unstable", delta_color="inverse")
                
                # Main visualization and data columns
                viz_col, data_col = st.columns([2, 1])
                
                with viz_col:
                    st.markdown("#### Real-time Parameters")
                    fig = visualizer.plot_data(data, faults, detector.thresholds, predictions)
                    st.pyplot(fig)
                
                with data_col:
                    st.markdown("#### Recent Alerts")
                    
                    if recent_faults:
                        for fault in recent_faults[:10]:
                            severity = fault.get('severity', 'info')
                            alert_class = f"{severity}-alert"
                            
                            st.markdown(f"""
                            <div class="{alert_class}">
                                <strong>{fault['fault_type']}</strong><br>
                                <small>Time: {fault['timestamp']}</small><br>
                                <small>{fault['parameter']}: {fault['value']}</small><br>
                                <small>Section: {fault['grid_section']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("No recent alerts")
                
                # Predictions section
                if predictions and show_predictions:
                    st.markdown("#### Predictive Analysis")
                    pred_cols = st.columns(len(predictions))
                    
                    for i, pred in enumerate(predictions):
                        with pred_cols[i]:
                            confidence_level = "HIGH" if pred['confidence'] > 70 else "MEDIUM" if pred['confidence'] > 40 else "LOW"
                            st.markdown(f"""
                            **[{confidence_level}] {pred['type']}**  
                            Confidence: {pred['confidence']:.1f}%  
                            ETA: {pred['estimated_time']}
                            """)
                
                # System status
                st.markdown("#### System Status")
                status_col1, status_col2, status_col3 = st.columns(3)
                
                with status_col1:
                    st.success("Data Stream: Active")
                
                with status_col2:
                    alert_status = "Normal" if stats['critical_last_hour'] == 0 else "Active Alerts"
                    if stats['critical_last_hour'] == 0:
                        st.success(f"Alert System: {alert_status}")
                    else:
                        st.error(f"Alert System: {alert_status}")
                
                with status_col3:
                    st.info(f"Monitoring: {selected_section}")

        except FileNotFoundError:
            with placeholder.container():
                st.warning("Waiting for data stream... Please run producer.py in a separate terminal.")
        except pd.errors.EmptyDataError:
            with placeholder.container():
                st.warning("Data file is empty, waiting for data...")
        except Exception as e:
            with placeholder.container():
                st.error(f"An error occurred: {e}")

        time.sleep(refresh_interval)

elif page == "Analytics":
    st.title("Grid Analytics Dashboard")
    
    try:
        # Get historical data
        recent_faults = detector.get_recent_faults(100)
        
        if recent_faults:
            # Severity Distribution
            st.markdown("#### Severity Distribution")
            pie_fig = visualizer.plot_severity_distribution(recent_faults)
            if pie_fig:
                st.pyplot(pie_fig)
            
            # Trend analysis
            st.markdown("#### Trend Analysis")
            if st.button("Generate Trend Analysis"):
                try:
                    data = pd.read_csv('live_data.csv')
                    trend_fig = visualizer.plot_trend_analysis(data)
                    if trend_fig:
                        st.pyplot(trend_fig)
                except:
                    st.warning("Unable to generate trend analysis - insufficient data")
            
        else:
            st.info("No historical data available for analysis")
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

elif page == "Historical Data":
    st.title("Historical Data Management")
    
    # Data management controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Cleanup Old Data"):
            deleted_count = detector.cleanup_old_data()
            st.success(f"Cleaned up {deleted_count} old records")
    
    with col2:
        retention_days = st.number_input("Retention Period (days)", 1, 365, 
                                       config['data_settings']['retention_days'])
    
    with col3:
        if st.button("Export Data"):
            recent_faults = detector.get_recent_faults(1000)
            if recent_faults:
                df = pd.DataFrame(recent_faults)
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "fault_history.csv", "text/csv")
    
    # Display recent data
    st.markdown("#### Recent Fault Records")
    recent_faults = detector.get_recent_faults(50)
    
    if recent_faults:
        df = pd.DataFrame(recent_faults)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No historical data available")

elif page == "System Settings":
    st.title("System Configuration")
    
    # Display current configuration
    st.markdown("#### Current Configuration")
    
    # Fault thresholds
    st.markdown("##### Fault Thresholds")
    threshold_data = []
    for fault_type, settings in config['fault_thresholds'].items():
        threshold_data.append({
            'Fault Type': fault_type.replace('_', ' ').title(),
            'Threshold': str(settings),
            'Severity': settings.get('severity', 'info')
        })
    
    st.dataframe(pd.DataFrame(threshold_data), use_container_width=True)
    
    # Alert settings
    st.markdown("##### Alert Configuration")
    alert_config = config['alert_settings']
    
    col1, col2 = st.columns(2)
    with col1:
        st.json(alert_config['email_alerts'])
    with col2:
        st.json(alert_config['sms_alerts'])
    
    # System status
    st.markdown("##### System Information")
    system_info = {
        'Configuration File': 'config.json',
        'Database': 'alerts.db',
        'Data Retention': f"{config['data_settings']['retention_days']} days",
        'Refresh Interval': f"{refresh_interval} seconds"
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")
    
    # Test alert system
    if st.button("Test Alert System"):
        test_fault = {
            'timestamp': datetime.now().isoformat(),
            'fault_type': 'Test Alert',
            'parameter': 'System Test',
            'value': 'Test Value',
            'grid_section': selected_section
        }
        
        try:
            processed = detector.alert_manager.process_fault(test_fault)
            st.success("Test alert processed successfully!")
            st.json(processed)
        except Exception as e:
            st.error(f"Test alert failed: {e}")

# Footer
st.markdown("---")
st.markdown("**Smart Grid Monitoring System** | Real-time fault detection with predictive analytics")