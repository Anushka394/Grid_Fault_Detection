import matplotlib.pyplot as plt

class DataVisualizer:
    def plot_data(self, data, faults, thresholds):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Smart Grid Parameters Analysis', fontsize=16)

        # Voltage Plot
        axs[0, 0].plot(data['Timestamp'], data['Voltage(V)'], label='Voltage (V)', color='blue')
        axs[0, 0].axhline(thresholds['under_voltage']['max_voltage'], color='orange', ls='--', label='UV Threshold')
        axs[0, 0].set_title('Voltage vs Time')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        # Current Plot
        axs[0, 1].plot(data['Timestamp'], data['Current(A)'], label='Current (A)', color='green')
        axs[0, 1].axhline(thresholds['over_current']['min_current'], color='red', ls='--', label='OC Threshold')
        axs[0, 1].set_title('Current vs Time')
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        # Frequency Plot
        axs[1, 0].plot(data['Timestamp'], data['Frequency(Hz)'], label='Frequency (Hz)', color='purple')
        axs[1, 0].axhline(thresholds['under_frequency']['min_freq'], color='red', ls='--', label='UF Threshold')
        axs[1, 0].axhline(thresholds['over_frequency']['max_freq'], color='red', ls='--', label='OF Threshold')
        axs[1, 0].set_title('Frequency vs Time')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # Power Factor Plot
        axs[1, 1].plot(data['Timestamp'], data['PowerFactor'], label='Power Factor', color='brown')
        axs[1, 1].axhline(thresholds['low_power_factor']['min_pf'], color='red', ls='--', label='LPF Threshold')
        axs[1, 1].set_title('Power Factor vs Time')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        if faults:
            for t, f_type, param, val in faults:
                if 'Voltage' in param:
                    axs[0, 0].plot(t, data.loc[data['Timestamp'] == t, 'Voltage(V)'], 'ro')
                elif 'Current' in param:
                    axs[0, 1].plot(t, data.loc[data['Timestamp'] == t, 'Current(A)'], 'ro')
                elif 'Frequency' in param:
                    axs[1, 0].plot(t, data.loc[data['Timestamp'] == t, 'Frequency(Hz)'], 'ro')
                elif 'Power Factor' in param:
                    axs[1, 1].plot(t, data.loc[data['Timestamp'] == t, 'PowerFactor'], 'ro')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig