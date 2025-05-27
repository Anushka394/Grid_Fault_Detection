Smart Grid Fault Detection

This project simulates a basic smart grid fault detection system using Python. It detects common faults like under-voltage, overcurrent, and possible earth faults based on voltage and current data.

Features

- Fault detection logic based on threshold rules
- Visual plots for voltage and current vs time
- Prints fault types with timestamps

How to Run

1. Install dependencies:
bash
pip install pandas matplotlib

2. Run the project:
bash
cd src
python main.py


Files

- `data/grid_data.csv`: Simulated sensor data
- `main.py`: Main program
- `detector.py`: Fault detection logic
- `visualizer.py`: Plotting code
