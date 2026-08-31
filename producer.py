import pandas as pd
import time
import os

SOURCE_FILE = "grid_data.csv"
LIVE_FILE   = "live_data.csv"
DELAY       = 2  # seconds between rows

print("Smart Grid Data Producer")
print("=" * 40)


def produce_data():
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: '{SOURCE_FILE}' not found. Run setup.py to generate it.")
        return

    source = pd.read_csv(SOURCE_FILE)

    with open(LIVE_FILE, "w") as f:
        f.write(",".join(source.columns) + "\n")
    print(f"Created '{LIVE_FILE}' with header.")
    print("Streaming data — press Ctrl+C to stop.\n")

    index = 0
    while True:
        row = source.iloc[[index]]
        with open(LIVE_FILE, "a") as f:
            f.write(row.to_csv(header=False, index=False))

        v = row.iloc[0]["Voltage(V)"]
        i = row.iloc[0]["Current(A)"]
        t = row.iloc[0]["Timestamp"]
        status = "NORMAL" if v > 200 and i < 12 else "WARNING" if v > 180 else "CRITICAL"
        print(f"[{status}] T:{t} | V:{v:.1f}V | I:{i:.1f}A")

        index = (index + 1) % len(source)
        time.sleep(DELAY)


if __name__ == "__main__":
    try:
        produce_data()
    except FileNotFoundError:
        print(f"Error: '{SOURCE_FILE}' not found. Run setup.py first.")
    except KeyboardInterrupt:
        print("\nProducer stopped.")
        if os.path.exists(LIVE_FILE):
            os.remove(LIVE_FILE)
        print(f"Removed '{LIVE_FILE}'.")
