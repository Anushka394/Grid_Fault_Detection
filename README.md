# Smart Grid Fault Detection Dashboard

This project simulates a live monitoring system for a smart grid. It detects common electrical faults like under-voltage, over-current, and frequency issues from a simulated live data stream and displays the results in a real-time web dashboard.

## Features

- Real-time fault detection from a live data stream.
- Detects multiple fault types: Voltage, Current, Frequency, and Power Factor.
- Interactive web dashboard built with Streamlit that updates automatically.
- Visual graphs for all key grid parameters.
- A live log of all detected faults.
- All fault thresholds can be easily changed in the `config.json` file.

## How to Run

1.  **Setup the Environment**
    - Open your terminal in the project folder.
    - Install the required libraries:
      ```bash
      pip install streamlit pandas matplotlib
      ```

2.  **Run the System**
    - This system needs two terminals running at the same time.

    - **In Terminal 1**, start the data producer. This script simulates live data.
      ```bash
      python producer.py
      ```

    - **In Terminal 2**, start the dashboard application.
      ```bash
      streamlit run dashboard.py
      ```
    - Your web browser will automatically open with the live dashboard.

## Files in the Project

- **`dashboard.py`**: The main Streamlit web application.
- **`producer.py`**: Simulates live data by sending readings every few seconds.
- **`detector.py`**: Contains the core logic for finding faults in the data.
- **`visualizer.py`**: Contains the code to create the graphs for the dashboard.
- **`config.json`**: All settings and fault limits are stored here.
- **`grid_data.csv`**: The source data file used by the producer script.
