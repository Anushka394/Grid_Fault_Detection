import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class EnhancedDataVisualizer:
    def __init__(self, theme="light"):
        self.theme = theme
        self._setup_style()

    def _setup_style(self):
        plt.style.use("default" if self.theme == "light" else "dark_background")
        self.bg_color   = "white" if self.theme == "light" else "#2E2E2E"
        self.text_color = "black" if self.theme == "light" else "white"
        sns.set_palette("husl")

    def plot_data(self, data, faults, thresholds, predictions=None):
        """Four-panel real-time parameter plot with fault markers and prediction overlays."""
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Smart Grid Parameters", fontsize=18, fontweight="bold")

        severity_colors = {"critical": "#FF4444", "warning": "#FFA500", "info": "#4444FF"}

        # Voltage
        axs[0, 0].plot(data["Timestamp"], data["Voltage(V)"], color="#2E86AB", linewidth=2, label="Voltage (V)")
        axs[0, 0].axhline(thresholds["under_voltage"]["max_voltage"],
                          color="orange", ls="--", alpha=0.7, label="UV Threshold")
        axs[0, 0].axhspan(thresholds["voltage_sag"]["min_voltage"],
                          thresholds["voltage_sag"]["max_voltage"],
                          alpha=0.2, color="yellow", label="Sag Range")
        self._add_fault_markers(axs[0, 0], data, faults, "Voltage", severity_colors)
        axs[0, 0].set_title("Voltage vs Time", fontsize=14, fontweight="bold")
        axs[0, 0].set_ylabel("Voltage (V)")
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        # Current
        axs[0, 1].plot(data["Timestamp"], data["Current(A)"], color="#A23B72", linewidth=2, label="Current (A)")
        axs[0, 1].axhline(thresholds["over_current"]["min_current"],
                          color="red", ls="--", alpha=0.7, label="OC Threshold")
        self._add_fault_markers(axs[0, 1], data, faults, "Current", severity_colors)
        axs[0, 1].set_title("Current vs Time", fontsize=14, fontweight="bold")
        axs[0, 1].set_ylabel("Current (A)")
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()

        # Frequency
        axs[1, 0].plot(data["Timestamp"], data["Frequency(Hz)"], color="#F18F01", linewidth=2, label="Frequency (Hz)")
        axs[1, 0].axhline(thresholds["under_frequency"]["min_freq"],
                          color="red", ls="--", alpha=0.7, label="UF Threshold")
        axs[1, 0].axhline(thresholds["over_frequency"]["max_freq"],
                          color="red", ls="--", alpha=0.7, label="OF Threshold")
        self._add_fault_markers(axs[1, 0], data, faults, "Frequency", severity_colors)
        axs[1, 0].set_title("Frequency vs Time", fontsize=14, fontweight="bold")
        axs[1, 0].set_ylabel("Frequency (Hz)")
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()

        # Power Factor
        axs[1, 1].plot(data["Timestamp"], data["PowerFactor"], color="#C73E1D", linewidth=2, label="Power Factor")
        axs[1, 1].axhline(thresholds["low_power_factor"]["min_pf"],
                          color="red", ls="--", alpha=0.7, label="LPF Threshold")
        self._add_fault_markers(axs[1, 1], data, faults, "Power Factor", severity_colors)
        axs[1, 1].set_title("Power Factor vs Time", fontsize=14, fontweight="bold")
        axs[1, 1].set_ylabel("Power Factor")
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend()

        if predictions:
            self._add_predictions(axs, predictions)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def _add_fault_markers(self, ax, data, faults, parameter, severity_colors):
        """Overlay fault event markers on a parameter axis."""
        if not faults:
            return
        for fault in faults:
            t, _, param, _, severity = (fault[:5] if len(fault) >= 5 else (*fault[:4], "warning"))
            if parameter not in param:
                continue
            match = data[data["Timestamp"] == t]
            if match.empty:
                continue
            col_map = {
                "Voltage":      "Voltage(V)",
                "Current":      "Current(A)",
                "Frequency":    "Frequency(Hz)",
                "Power Factor": "PowerFactor",
            }
            if parameter not in col_map:
                continue
            y_val = match[col_map[parameter]].iloc[0]
            ax.plot(t, y_val, "o", color=severity_colors.get(severity, "#FF4444"),
                    markersize=8, markeredgecolor="black", markeredgewidth=1)

    def _add_predictions(self, axs, predictions):
        """Overlay prediction confidence text on the relevant parameter axis."""
        ax_map = {"Voltage": axs[0, 0], "Current": axs[0, 1], "Frequency": axs[1, 0]}
        for pred in predictions:
            ax = ax_map.get(pred["parameter"])
            if ax is None:
                continue
            ax.text(
                0.02, 0.98,
                f"WARN: {pred['type']}\nConfidence: {pred['confidence']:.1f}%",
                transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                fontsize=9,
            )

    def plot_severity_distribution(self, fault_data):
        """Pie chart of fault counts by severity."""
        if not fault_data:
            return None
        df = pd.DataFrame(fault_data)
        counts = df["severity"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=["#FF4444", "#FFA500", "#4444FF"], startangle=90,
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
        ax.set_title("Fault Distribution by Severity", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_trend_analysis(self, data, window=10):
        """Four-panel trend analysis with raw data, moving average, and linear trend line."""
        if len(data) < window:
            return None
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Trend Analysis with Moving Averages", fontsize=16, fontweight="bold")

        params = [
            ("Voltage(V)",    "Voltage (V)",    axs[0, 0]),
            ("Current(A)",    "Current (A)",    axs[0, 1]),
            ("Frequency(Hz)", "Frequency (Hz)", axs[1, 0]),
            ("PowerFactor",   "Power Factor",   axs[1, 1]),
        ]
        for col, label, ax in params:
            ax.plot(data["Timestamp"], data[col], alpha=0.5, label="Raw")
            ax.plot(data["Timestamp"], data[col].rolling(window=window).mean(),
                    linewidth=2, label=f"{window}-pt MA")
            if len(data) > 1:
                x = range(len(data))
                z = np.polyfit(x, data[col], 1)
                ax.plot(data["Timestamp"], np.poly1d(z)(x), "--",
                        label=f"Trend (slope: {z[0]:.3f})")
            ax.set_title(f"{label} Trend")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def plot_fault_heatmap(self, fault_data):
        """Heatmap of fault frequency by hour and fault type."""
        if not fault_data:
            return None
        df = pd.DataFrame(fault_data)
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        heatmap_data = df.groupby(["hour", "fault_type"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, cmap="Reds", ax=ax, fmt="d")
        ax.set_title("Fault Frequency Heatmap (by Hour)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Fault Type")
        ax.set_ylabel("Hour of Day")
        plt.tight_layout()
        return fig

    def plot_grid_topology(self, grid_sections):
        """Simple node-connection diagram of the grid sections."""
        fig, ax = plt.subplots(figsize=(10, 6))
        positions = {
            "Substation_A": (0.2, 0.7),
            "Substation_B": (0.8, 0.7),
            "Load_Center":  (0.5, 0.3),
        }
        for start, end in [("Substation_A", "Load_Center"), ("Substation_B", "Load_Center")]:
            if start in positions and end in positions:
                xs = [positions[start][0], positions[end][0]]
                ys = [positions[start][1], positions[end][1]]
                ax.plot(xs, ys, "k-", linewidth=3, alpha=0.7)
        for section, (x, y) in positions.items():
            if section in grid_sections:
                priority = grid_sections[section].get("priority", "medium")
                color = {"high": "red", "medium": "orange", "low": "green"}.get(priority, "blue")
                ax.scatter(x, y, s=1000, c=color, alpha=0.7, edgecolors="black", linewidth=2)
                ax.text(x, y - 0.10, section.replace("_", " "), ha="center", fontweight="bold")
                ax.text(x, y - 0.15, f"Priority: {priority}", ha="center", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Grid Topology", fontsize=16, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        return fig
