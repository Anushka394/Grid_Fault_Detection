import matplotlib.pyplot as plt

def plot_data(data):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(data['Timestamp'], data['Voltage(V)'], label='Voltage (V)', color='blue')
    plt.axhline(100, color='red', linestyle='--', label='Low Voltage Threshold')
    plt.legend()
    plt.title("Voltage vs Time")

    plt.subplot(2, 1, 2)
    plt.plot(data['Timestamp'], data['Current(A)'], label='Current (A)', color='green')
    plt.axhline(15, color='red', linestyle='--', label='Overcurrent Threshold')
    plt.legend()
    plt.title("Current vs Time")

    plt.tight_layout()
    plt.show()
