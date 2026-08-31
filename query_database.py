import sqlite3
import pandas as pd

conn = sqlite3.connect("alerts.db")

print("=" * 60)
print("ALL ALERTS")
print("=" * 60)
df = pd.read_sql_query("SELECT * FROM alerts", conn)
print(df)
print(f"\nTotal: {len(df)}")

print("\n" + "=" * 60)
print("BY SEVERITY")
print("=" * 60)
print(pd.read_sql_query(
    "SELECT severity, COUNT(*) as count FROM alerts GROUP BY severity", conn
))

print("\n" + "=" * 60)
print("RECENT 10")
print("=" * 60)
print(pd.read_sql_query(
    "SELECT timestamp, fault_type, severity, parameter, value "
    "FROM alerts ORDER BY created_at DESC LIMIT 10", conn
))

print("\n" + "=" * 60)
print("CRITICAL ALERTS")
print("=" * 60)
critical = pd.read_sql_query(
    "SELECT timestamp, fault_type, parameter, value "
    "FROM alerts WHERE severity = 'critical'", conn
)
print(critical)
print(f"\nTotal critical: {len(critical)}")

print("\n" + "=" * 60)
print("BY FAULT TYPE")
print("=" * 60)
print(pd.read_sql_query(
    "SELECT fault_type, COUNT(*) as count FROM alerts "
    "GROUP BY fault_type ORDER BY count DESC", conn
))

conn.close()
print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
