import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('alerts.db')

print("=" * 60)
print("ALL ALERTS:")
print("=" * 60)
df = pd.read_sql_query("SELECT * FROM alerts", conn)
print(df)
print(f"\nTotal Alerts: {len(df)}")

# Query 2: Count by severity
print("\n" + "=" * 60)
print("ALERTS BY SEVERITY:")
print("=" * 60)
severity_df = pd.read_sql_query("""
    SELECT severity, COUNT(*) as count 
    FROM alerts 
    GROUP BY severity
""", conn)
print(severity_df)

# Query 3: Recent 10 alerts
print("\n" + "=" * 60)
print("RECENT 10 ALERTS:")
print("=" * 60)
recent_df = pd.read_sql_query("""
    SELECT timestamp, fault_type, severity, parameter, value 
    FROM alerts 
    ORDER BY created_at DESC 
    LIMIT 10
""", conn)
print(recent_df)

# Query 4: Critical alerts only
print("\n" + "=" * 60)
print("CRITICAL ALERTS:")
print("=" * 60)
critical_df = pd.read_sql_query("""
    SELECT timestamp, fault_type, parameter, value 
    FROM alerts 
    WHERE severity = 'critical'
""", conn)
print(critical_df)
print(f"\nTotal Critical Alerts: {len(critical_df)}")

# Query 5: Alerts by fault type
print("\n" + "=" * 60)
print("ALERTS BY FAULT TYPE:")
print("=" * 60)
fault_type_df = pd.read_sql_query("""
    SELECT fault_type, COUNT(*) as count 
    FROM alerts 
    GROUP BY fault_type 
    ORDER BY count DESC
""", conn)
print(fault_type_df)

conn.close()
print("\n" + "=" * 60)
print("Database query completed!")
print("=" * 60)
