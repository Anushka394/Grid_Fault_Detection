import pandas as pd
import time
import os

SOURCE_FILE = 'grid_data.csv'
LIVE_FILE = 'live_data.csv'
DELAY_SECONDS = 2

print("Smart Grid Data Producer")
print("=" * 40) 

def produce_data():
    """Reads from a source CSV and writes to a live CSV line by line."""
    print("Starting data producer...")
    
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: Source file '{SOURCE_FILE}' not found.")
        print("Run 'python setup.py' to create sample data")
        return
    
    source_df = pd.read_csv(SOURCE_FILE)
    
    header = ",".join(source_df.columns) + '\n'
    with open(LIVE_FILE, 'w') as f:
        f.write(header)
    print(f"Live data file '{LIVE_FILE}' created with header.")

    index = 0
    print("Streaming data to monitoring system...")
    print("Press Ctrl+C to stop")
    
    while True:
        row = source_df.iloc[[index]]
        row_csv = row.to_csv(header=False, index=False)
        
        with open(LIVE_FILE, 'a') as f:
            f.write(row_csv)
            
        timestamp = row.iloc[0]['Timestamp']
        voltage = row.iloc[0]['Voltage(V)']
        current = row.iloc[0]['Current(A)']
        
        # Status display
        status = "NORMAL" if voltage > 200 and current < 12 else "WARNING" if voltage > 180 else "CRITICAL"
        print(f"[{status}] T:{timestamp} | V:{voltage:.1f}V | I:{current:.1f}A | Data streamed")
        
        index = (index + 1) % len(source_df) 
        time.sleep(DELAY_SECONDS)

if __name__ == "__main__":
    try:
        produce_data()
    except FileNotFoundError:
        print(f"Error: Source file '{SOURCE_FILE}' not found.")
        print("Run 'python setup.py' to create sample data")
    except KeyboardInterrupt:
        print("\nProducer stopped by user.")
        if os.path.exists(LIVE_FILE):
            os.remove(LIVE_FILE)
        print(f"Cleaned up '{LIVE_FILE}'.")
        print("Data producer shutdown complete.")