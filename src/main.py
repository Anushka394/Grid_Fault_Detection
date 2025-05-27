# main.py (located in src/)

import pandas as pd
from detector import detect_faults
from visualizer import plot_data

def main():
    data = pd.read_csv('../data/grid_data.csv')
    
    faults = detect_faults(data)
    for t, f in faults:
        print(f"Time {t}: {f}")
    
    plot_data(data, faults)

if __name__ == "__main__":
    main()
