# Smart Grid Monitoring System

A real-time smart grid fault detection and monitoring system with advanced analytics and alerting capabilities.

## What is this project?

This system monitors electrical grid parameters (voltage, current, frequency, power factor) in real-time and automatically detects various types of electrical faults. It provides:

- **Real-time monitoring** of grid parameters
- **Automatic fault detection** for 7+ types of electrical faults
- **Predictive analytics** to forecast potential issues
- **Email alerts** for critical faults
- **Interactive dashboard** with multiple pages
- **Historical data analysis** and reporting

## Fault Types Detected

1. **Earth Fault** - Low voltage + high current (Critical)
2. **Under-voltage** - Voltage below safe limits (Warning)
3. **Overcurrent** - Current above safe limits (Critical)
4. **Under/Over Frequency** - Frequency deviation (Warning)
5. **Low Power Factor** - Poor power quality (Info)
6. **Voltage Sag** - Temporary voltage reduction (Warning)
7. **Harmonic Distortion** - Power quality issues (Info)

## Quick Start

1. **Setup** (one-time):
   ```bash
   python setup.py
   ```

2. **Start data stream**:
   ```bash
   python producer.py
   ```

3. **Launch dashboard** (new terminal):
   ```bash
   streamlit run dashboard.py
   ```

4. **Access dashboard**: http://localhost:8501

## System Components

- `dashboard.py` - Multi-page monitoring interface
- `detector.py` - Fault detection engine
- `visualizer.py` - Data visualization
- `alert_manager.py` - Email alerts & database logging
- `producer.py` - Data stream simulator
- `config.json` - System configuration

## Use Cases

- Power grid monitoring
- Electrical fault detection
- Predictive maintenance
- Power quality analysis
- Grid operator dashboards

Built with Python, Streamlit, and advanced analytics for professional grid monitoring applications.