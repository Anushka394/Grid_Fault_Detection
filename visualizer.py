import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class EnhancedDataVisualizer:
    def __init__(self, theme="light"):
        self.theme = theme
        self._setup_style()

    def _setup_style(self):
        plt.style.use("dark_background")
        self.bg_color   = "#161b27"
        self.text_color = "#e0e0e0"
        sns.set_palette("husl")

    def plot_data(self, data, faults, thresholds, predictions=None):
        """Four-panel real-time parameter plot with fault markers and prediction overlays."""
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_facecolor("#161b27")
        fig.suptitle("Real-time Grid Parameters", fontsize=15, fontweight="bold",
                     color="#e0e0e0", y=1.01)

        severity_colors = {"critical": "#ef5350", "warning": "#ffa726", "info": "#29b6f6"}

        panels = [
            (axs[0, 0], "Voltage(V)",    "Voltage (V)",    "#42a5f5"),
            (axs[0, 1], "Current(A)",    "Current (A)",    "#ab47bc"),
            (axs[1, 0], "Frequency(Hz)", "Frequency (Hz)", "#ffa726"),
            (axs[1, 1], "PowerFactor",   "Power Factor",   "#ef5350"),
        ]

        threshold_lines = {
            "Voltage(V)": [
                (thresholds["under_voltage"]["max_voltage"], "#ffa726", "UV Limit"),
            ],
            "Current(A)": [
                (thresholds["over_current"]["min_current"], "#ef5350", "OC Limit"),
            ],
            "Frequency(Hz)": [
                (thresholds["under_frequency"]["min_freq"], "#ef5350", "UF Limit"),
                (thresholds["over_frequency"]["max_freq"],  "#ef5350", "OF Limit"),
            ],
            "PowerFactor": [
                (thresholds["low_power_factor"]["min_pf"], "#ef5350", "LPF Limit"),
            ],
        }

        for ax, col, label, color in panels:
            ax.set_facecolor("#1c2133")
            ax.tick_params(colors="#8b9ab0", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2f45")

            ax.plot(data["Timestamp"], data[col], color=color, linewidth=1.8,
                    label=label, zorder=3)

            for y_val, lc, lname in threshold_lines.get(col, []):
                ax.axhline(y_val, color=lc, ls="--", alpha=0.6, linewidth=1, label=lname)

            if col == "Voltage(V)" and "voltage_sag" in thresholds:
                ax.axhspan(thresholds["voltage_sag"]["min_voltage"],
                           thresholds["voltage_sag"]["max_voltage"],
                           alpha=0.12, color="#ffd54f", label="Sag Zone")

            self._add_fault_markers(ax, data, faults,
                                    label.replace(" (V)", "").replace(" (A)", "")
                                         .replace(" (Hz)", "").replace(" ", " "),
                                    severity_colors)

            ax.set_title(label, fontsize=11, fontweight="bold", color="#c9d1e0", pad=8)
            ax.set_ylabel(label, fontsize=8, color="#8b9ab0")
            ax.grid(True, alpha=0.15, color="#2a2f45", linestyle="--")
            legend = ax.legend(fontsize=7, loc="upper right",
                               facecolor="#1c2133", edgecolor="#2a2f45",
                               labelcolor="#8b9ab0")

        if predictions:
            self._add_predictions(axs, predictions)

        plt.tight_layout()
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
            c = pred["confidence"]
            fc = "#3a1a1f" if c > 70 else "#2a2010" if c > 40 else "#1b3a2a"
            tc = "#ef5350"  if c > 70 else "#ffa726" if c > 40 else "#66bb6a"
            ax.text(
                0.02, 0.97,
                f"WARN  {c:.0f}%\n{pred['type']}",
                transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=fc, edgecolor=tc, alpha=0.9),
                fontsize=8, color=tc, fontweight="bold",
            )

    def plot_severity_distribution(self, fault_data):
        """Pie chart of fault counts by severity."""
        if not fault_data:
            return None
        df = pd.DataFrame(fault_data)
        counts = df["severity"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#161b27")
        ax.set_facecolor("#161b27")
        color_map = {"critical": "#ef5350", "warning": "#ffa726", "info": "#29b6f6"}
        colors = [color_map.get(s, "#8b9ab0") for s in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90,
            textprops={"color": "#c9d1e0", "fontsize": 10},
        )
        for at in autotexts:
            at.set_color("#0f1117")
            at.set_fontweight("bold")
            at.set_fontsize(9)
        ax.set_title("Fault Distribution by Severity", fontsize=12,
                     fontweight="bold", color="#e0e0e0", pad=12)
        plt.tight_layout()
        return fig

    def plot_trend_analysis(self, data, window=10):
        """Four-panel trend analysis with raw data, moving average, and linear trend line."""
        if len(data) < window:
            return None
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        fig.patch.set_facecolor("#161b27")
        fig.suptitle("Trend Analysis with Moving Averages", fontsize=14,
                     fontweight="bold", color="#e0e0e0")

        params = [
            ("Voltage(V)",    "Voltage (V)",    axs[0, 0], "#42a5f5"),
            ("Current(A)",    "Current (A)",    axs[0, 1], "#ab47bc"),
            ("Frequency(Hz)", "Frequency (Hz)", axs[1, 0], "#ffa726"),
            ("PowerFactor",   "Power Factor",   axs[1, 1], "#ef5350"),
        ]
        for col, label, ax, color in params:
            ax.set_facecolor("#1c2133")
            ax.tick_params(colors="#8b9ab0", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2f45")
            ax.plot(data["Timestamp"], data[col], alpha=0.35, color=color, label="Raw")
            ax.plot(data["Timestamp"], data[col].rolling(window=window).mean(),
                    linewidth=2, color=color, label=f"{window}-pt MA")
            if len(data) > 1:
                x = range(len(data))
                z = np.polyfit(x, data[col], 1)
                ax.plot(data["Timestamp"], np.poly1d(z)(x), "--",
                        color="#8b9ab0", linewidth=1, label=f"Trend ({z[0]:.3f})")
            ax.set_title(f"{label} Trend", fontsize=10, fontweight="bold", color="#c9d1e0")
            ax.grid(True, alpha=0.15, color="#2a2f45", linestyle="--")
            ax.legend(fontsize=7, facecolor="#1c2133", edgecolor="#2a2f45", labelcolor="#8b9ab0")

        plt.tight_layout()
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
