#!/usr/bin/env python3


import os
import sqlite3
import json
from datetime import datetime

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'exports', 'backups']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def initialize_database():
    """Initialize the alerts database"""
    try:
        conn = sqlite3.connect('alerts.db')
        cursor = conn.cursor()
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                fault_type TEXT,
                severity TEXT,
                parameter TEXT,
                value TEXT,
                grid_section TEXT,
                alert_sent BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON alerts(severity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON alerts(created_at)')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Database initialization failed: {e}")

def validate_config():
    """Validate configuration files"""
    config_files = ['config.json']
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
                print(f"Configuration file valid: {config_file}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {config_file}: {e}")
        else:
            print(f"Configuration file not found: {config_file}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'matplotlib', 'seaborn', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"{package} is missing")
    
    if missing_packages:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def create_sample_data():
    """Create sample data if grid_data.csv doesn't exist"""
    if not os.path.exists('grid_data.csv'):
        print("Creating sample grid data...")
        
        import pandas as pd
        import numpy as np
        
        # Generate sample data
        timestamps = range(100)
        np.random.seed(42)
        
        data = {
            'Timestamp': timestamps,
            'Voltage(V)': np.random.normal(230, 10, 100),
            'Current(A)': np.random.normal(8, 2, 100),
            'Frequency(Hz)': np.random.normal(50, 0.2, 100),
            'PowerFactor': np.random.normal(0.95, 0.05, 100)
        }
        
        # Add some fault conditions
        data['Voltage(V)'][20:25] = 90  # Earth fault simulation
        data['Current(A)'][20:25] = 18
        data['Voltage(V)'][50:55] = 170  # Under-voltage
        data['Frequency(Hz)'][80:85] = 48.5  # Under-frequency
        
        df = pd.DataFrame(data)
        df.to_csv('grid_data.csv', index=False)
        print("Sample grid data created: grid_data.csv")

def main():
    """Main setup function"""
    print("Setting up Smart Grid Monitoring System...")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies before continuing")
        return
    
    # Create directories
    create_directories()
    
    # Initialize database
    initialize_database()
    
    # Validate configuration
    validate_config()
    
    # Create sample data if needed
    create_sample_data()
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Review and update config.json for your environment")
    print("2. Configure email settings if you want alerts")
    print("3. Run: python producer.py (in one terminal)")
    print("4. Run: streamlit run dashboard.py (in another terminal)")
    print("\nThe dashboard will be available at: http://localhost:8501")
    print("\nSee README.md for detailed documentation")

if __name__ == "__main__":
    main()
