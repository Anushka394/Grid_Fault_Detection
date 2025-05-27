import pandas as pd

def detect_faults(data):
    faults = []
    for i, row in data.iterrows():
        if row['Voltage(V)'] < 100 and row['Current(A)'] > 10:
            faults.append((row['Timestamp'], "Possible Earth Fault"))
        elif row['Voltage(V)'] < 180:
            faults.append((row['Timestamp'], "Under-voltage"))
        elif row['Current(A)'] > 15:
            faults.append((row['Timestamp'], "Overcurrent"))
    return faults
