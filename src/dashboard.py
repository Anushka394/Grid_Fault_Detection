import streamlit as st
import pandas as pd
from detector import FaultDetector
from visualizer import DataVisualizer
import time

st.set_page_config(page_title="Live Smart Grid Monitoring", layout="wide")

st.title("Live Smart Grid Fault Detection")

placeholder = st.empty()

detector = FaultDetector(config_path='config.json')
visualizer = DataVisualizer()

while True:
    try:
        data = pd.read_csv('live_data.csv')
        
        data.drop_duplicates(subset=['Timestamp'], keep='last', inplace=True)
        
        faults = detector.detect_faults(data)
        faults_df = pd.DataFrame(faults, columns=['Timestamp', 'Fault Type', 'Parameter', 'Value'])
        
        with placeholder.container():
            st.header(f"Live Grid Status (Last updated: {time.strftime('%H:%M:%S')})")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Readings", len(data))
            col2.metric("Total Faults", len(faults_df), delta=f"{len(faults_df)} faults", delta_color="inverse")
            
            fig_col, data_col = st.columns(2)
            
            with fig_col:
                st.markdown("#### Data Visualization")
                fig = visualizer.plot_data(data, faults, detector.thresholds)
                st.pyplot(fig)

            with data_col:
                st.markdown("#### Live Faults Log")
                st.dataframe(faults_df.tail(10))

    except FileNotFoundError:
        with placeholder.container():
            st.warning("Waiting for data stream... Please run producer.py in a separate terminal.")
    except pd.errors.EmptyDataError:
        with placeholder.container():
            st.warning("Data file is empty, waiting for data...")
    except Exception as e:
        with placeholder.container():
            st.error(f"An error occurred: {e}")

    time.sleep(2)