import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EnhancedDataVisualizer:
    def __init__(self, theme='light'):
        self.theme = theme
        self.setup_style()
    
    def setup_style(self):
        """Setup matplotlib style based on theme"""
        if self.theme == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#2E2E2E'
            self.text_color = 'white'
        else:
            plt.style.use('default')
            self.bg_color = 'white'
            self.text_color = 'black'
        
        sns.set_palette("husl")
    
    def plot_data(self, data, faults, thresholds, predictions=None):
        """Enhanced data visualization with predictions and better styling"""
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Smart Grid Parameters Analysis', fontsize=18, fontweight='bold')
        
        # Color scheme for severity levels
        severity_colors = {
            'critical': '#FF4444',
            'warning': '#FFA500', 
            'info': '#4444FF'
        }
        
        # Voltage Plot
        axs[0, 0].plot(data['Timestamp'], data['Voltage(V)'], 
                      label='Voltage (V)', color='#2E86AB', linewidth=2)
        axs[0, 0].axhline(thresholds['under_voltage']['max_voltage'], 
                         color='orange', ls='--', alpha=0.7, label='UV Threshold')
        if 'voltage_sag' in thresholds:
            axs[0, 0].axhspan(thresholds['voltage_sag']['min_voltage'],
                             thresholds['voltage_sag']['max_voltage'],
                             alpha=0.2, color='yellow', label='Sag Range')
        
        self._add_fault_markers(axs[0, 0], data, faults, 'Voltage', severity_colors)
        axs[0, 0].set_title('Voltage vs Time', fontsize=14, fontweight='bold')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()
        axs[0, 0].set_ylabel('Voltage (V)')

        # Current Plot
        axs[0, 1].plot(data['Timestamp'], data['Current(A)'], 
                      label='Current (A)', color='#A23B72', linewidth=2)
        axs[0, 1].axhline(thresholds['over_current']['min_current'], 
                         color='red', ls='--', alpha=0.7, label='OC Threshold')
        
        self._add_fault_markers(axs[0, 1], data, faults, 'Current', severity_colors)
        axs[0, 1].set_title('Current vs Time', fontsize=14, fontweight='bold')
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()
        axs[0, 1].set_ylabel('Current (A)')

        # Frequency Plot
        axs[1, 0].plot(data['Timestamp'], data['Frequency(Hz)'], 
                      label='Frequency (Hz)', color='#F18F01', linewidth=2)
        axs[1, 0].axhline(thresholds['under_frequency']['min_freq'], 
                         color='red', ls='--', alpha=0.7, label='UF Threshold')
        axs[1, 0].axhline(thresholds['over_frequency']['max_freq'], 
                         color='red', ls='--', alpha=0.7, label='OF Threshold')
        
        self._add_fault_markers(axs[1, 0], data, faults, 'Frequency', severity_colors)
        axs[1, 0].set_title('Frequency vs Time', fontsize=14, fontweight='bold')
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()
        axs[1, 0].set_ylabel('Frequency (Hz)')

        # Power Factor Plot
        axs[1, 1].plot(data['Timestamp'], data['PowerFactor'], 
                      label='Power Factor', color='#C73E1D', linewidth=2)
        axs[1, 1].axhline(thresholds['low_power_factor']['min_pf'], 
                         color='red', ls='--', alpha=0.7, label='LPF Threshold')
        
        self._add_fault_markers(axs[1, 1], data, faults, 'Power Factor', severity_colors)
        axs[1, 1].set_title('Power Factor vs Time', fontsize=14, fontweight='bold')
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend()
        axs[1, 1].set_ylabel('Power Factor')

        # Add predictions if available
        if predictions:
            self._add_predictions(axs, predictions)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    
    def _add_fault_markers(self, ax, data, faults, parameter, severity_colors):
        """Add fault markers to plots with severity-based colors"""
        if not faults:
            return
            
        for fault in faults:
            if len(fault) >= 5:  # New format with severity
                t, f_type, param, val, severity = fault[:5]
            else:  # Old format compatibility
                t, f_type, param, val = fault[:4]
                severity = 'warning'
            
            if parameter in param:
                # Find the data point
                matching_rows = data[data['Timestamp'] == t]
                if not matching_rows.empty:
                    if parameter == 'Voltage':
                        y_val = matching_rows['Voltage(V)'].iloc[0]
                    elif parameter == 'Current':
                        y_val = matching_rows['Current(A)'].iloc[0]
                    elif parameter == 'Frequency':
                        y_val = matching_rows['Frequency(Hz)'].iloc[0]
                    elif parameter == 'Power Factor':
                        y_val = matching_rows['PowerFactor'].iloc[0]
                    else:
                        continue
                    
                    color = severity_colors.get(severity, '#FF4444')
                    ax.plot(t, y_val, 'o', color=color, markersize=8, 
                           markeredgecolor='black', markeredgewidth=1)
    
    def _add_predictions(self, axs, predictions):
        """Add prediction indicators to plots"""
        for pred in predictions:
            param = pred['parameter']
            confidence = pred['confidence']
            
            if param == 'Voltage':
                ax = axs[0, 0]
            elif param == 'Current':
                ax = axs[0, 1]
            elif param == 'Frequency':
                ax = axs[1, 0]
            else:
                continue
            
            # Add prediction indicator
            ax.text(0.02, 0.98, f"⚠️ {pred['type']}\nConfidence: {confidence:.1f}%", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   fontsize=9)
    
    def plot_fault_heatmap(self, fault_data):
        """Create a heatmap showing fault frequency by time and type"""
        if not fault_data:
            return None
        
        # Convert fault data to DataFrame
        df = pd.DataFrame(fault_data)
        
        # Create time bins (hourly)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Create pivot table for heatmap
        heatmap_data = df.groupby(['hour', 'fault_type']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='Reds', ax=ax, fmt='d')
        ax.set_title('Fault Frequency Heatmap (by Hour)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Fault Type')
        ax.set_ylabel('Hour of Day')
        
        plt.tight_layout()
        return fig
    
    def plot_severity_distribution(self, fault_data):
        """Create pie chart showing fault distribution by severity"""
        if not fault_data:
            return None
        
        df = pd.DataFrame(fault_data)
        severity_counts = df['severity'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#FF4444', '#FFA500', '#4444FF']
        
        wedges, texts, autotexts = ax.pie(severity_counts.values, 
                                         labels=severity_counts.index,
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90)
        
        ax.set_title('Fault Distribution by Severity', fontsize=14, fontweight='bold')
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        return fig
    
    def plot_trend_analysis(self, data, window=10):
        """Plot trend analysis with moving averages"""
        if len(data) < window:
            return None
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Trend Analysis with Moving Averages', fontsize=16, fontweight='bold')
        
        parameters = [
            ('Voltage(V)', 'Voltage (V)', axs[0, 0]),
            ('Current(A)', 'Current (A)', axs[0, 1]),
            ('Frequency(Hz)', 'Frequency (Hz)', axs[1, 0]),
            ('PowerFactor', 'Power Factor', axs[1, 1])
        ]
        
        for param, label, ax in parameters:
            # Original data
            ax.plot(data['Timestamp'], data[param], alpha=0.5, label='Raw Data')
            
            # Moving average
            ma = data[param].rolling(window=window).mean()
            ax.plot(data['Timestamp'], ma, linewidth=2, label=f'{window}-point MA')
            
            # Trend line
            if len(data) > 1:
                x_numeric = range(len(data))
                z = np.polyfit(x_numeric, data[param], 1)
                p = np.poly1d(z)
                ax.plot(data['Timestamp'], p(x_numeric), '--', 
                       label=f'Trend (slope: {z[0]:.3f})')
            
            ax.set_title(f'{label} Trend Analysis')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    
    def plot_grid_topology(self, grid_sections):
        """Simple grid topology visualization"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simple network diagram
        positions = {
            'Substation_A': (0.2, 0.7),
            'Substation_B': (0.8, 0.7),
            'Load_Center': (0.5, 0.3)
        }
        
        # Draw connections
        connections = [
            ('Substation_A', 'Load_Center'),
            ('Substation_B', 'Load_Center')
        ]
        
        for start, end in connections:
            if start in positions and end in positions:
                x_vals = [positions[start][0], positions[end][0]]
                y_vals = [positions[start][1], positions[end][1]]
                ax.plot(x_vals, y_vals, 'k-', linewidth=3, alpha=0.7)
        
        # Draw nodes
        for section, (x, y) in positions.items():
            if section in grid_sections:
                priority = grid_sections[section].get('priority', 'medium')
                color = {'high': 'red', 'medium': 'orange', 'low': 'green'}.get(priority, 'blue')
                ax.scatter(x, y, s=1000, c=color, alpha=0.7, edgecolors='black', linewidth=2)
                ax.text(x, y-0.1, section.replace('_', ' '), ha='center', fontweight='bold')
                ax.text(x, y-0.15, f"Priority: {priority}", ha='center', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Grid Topology Overview', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig