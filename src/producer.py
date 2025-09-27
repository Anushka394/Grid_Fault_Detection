import pandas as pd
import time
import os

SOURCE_FILE = 'grid_data.csv'
LIVE_FILE = 'live_data.csv'
DELAY_SECONDS = 2 

def produce_data():
    """Reads from a source CSV and writes to a live CSV line by line."""
    print("Starting data producer...")
    
    source_df = pd.read_csv(SOURCE_FILE)
    
    header = ",".join(source_df.columns) + '\n'
    with open(LIVE_FILE, 'w') as f:
        f.write(header)
    print(f"Live data file '{LIVE_FILE}' created with header.")

    index = 0
    while True:
        row = source_df.iloc[[index]]
        row_csv = row.to_csv(header=False, index=False)
        
        with open(LIVE_FILE, 'a') as f:
            f.write(row_csv)
            
        print(f"Timestamp {row.iloc[0]['Timestamp']}: Data produced.")
        
        index = (index + 1) % len(source_df) 
        time.sleep(DELAY_SECONDS)

if __name__ == "__main__":
    try:
        produce_data()
    except FileNotFoundError:
        print(f"Error: Source file '{SOURCE_FILE}' not found.")
    except KeyboardInterrupt:
        print("\nProducer stopped.")
        if os.path.exists(LIVE_FILE):
            os.remove(LIVE_FILE)
        print(f"Cleaned up '{LIVE_FILE}'.")