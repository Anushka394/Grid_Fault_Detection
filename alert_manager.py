import smtplib
import json
import sqlite3
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


class AlertManager:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.alert_settings = self.config["alert_settings"]
        self._setup_database()
        self._setup_logging()

    def _setup_database(self):
        self.conn = sqlite3.connect("alerts.db", check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT,
                fault_type   TEXT,
                severity     TEXT,
                parameter    TEXT,
                value        TEXT,
                grid_section TEXT,
                alert_sent   BOOLEAN,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _setup_logging(self):
        logging.basicConfig(
            filename="alert_system.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _should_send_alert(self, severity):
        return severity in self.alert_settings.get("alert_on_severity", [])

    def _send_email(self, fault_data):
        if not self.alert_settings["email_alerts"]["enabled"]:
            return False
        try:
            cfg = self.alert_settings["email_alerts"]
            msg = MIMEMultipart()
            msg["From"]    = cfg["sender_email"]
            msg["To"]      = ", ".join(cfg["recipients"])
            msg["Subject"] = f"GRID ALERT: {fault_data['fault_type']} - {fault_data['severity'].upper()}"
            body = (
                f"SMART GRID FAULT DETECTED\n\n"
                f"Timestamp    : {fault_data['timestamp']}\n"
                f"Fault Type   : {fault_data['fault_type']}\n"
                f"Severity     : {fault_data['severity'].upper()}\n"
                f"Parameter    : {fault_data['parameter']}\n"
                f"Value        : {fault_data['value']}\n"
                f"Grid Section : {fault_data.get('grid_section', 'Unknown')}\n\n"
                f"Grid Monitoring System"
            )
            msg.attach(MIMEText(body, "plain"))
            server = smtplib.SMTP(cfg["smtp_server"], cfg["smtp_port"])
            server.starttls()
            server.login(cfg["sender_email"], cfg["sender_password"])
            server.send_message(msg)
            server.quit()
            self.logger.info("Email alert sent for %s", fault_data["fault_type"])
            return True
        except Exception as e:
            self.logger.error("Failed to send email alert: %s", e)
            return False

    def _log_to_database(self, fault_data):
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts
                    (timestamp, fault_type, severity, parameter, value, grid_section, alert_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fault_data["timestamp"],
                    fault_data["fault_type"],
                    fault_data["severity"],
                    fault_data["parameter"],
                    fault_data["value"],
                    fault_data.get("grid_section", "Unknown"),
                    fault_data.get("alert_sent", False),
                ),
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error("Failed to log fault to database: %s", e)

    def process_fault(self, fault_data):
        """Assign severity, log to database, and dispatch email alert if warranted."""
        fault_key = fault_data["fault_type"].lower().replace("-", "_").replace(" ", "_")
        fault_data["severity"] = self._get_severity(fault_key)
        self._log_to_database(fault_data)
        if self._should_send_alert(fault_data["severity"]):
            fault_data["alert_sent"] = self._send_email(fault_data)
        return fault_data

    def _get_severity(self, fault_type):
        for key, data in self.config["fault_thresholds"].items():
            if fault_type in key or key in fault_type:
                return data.get("severity", "info")
        return "info"

    def get_recent_alerts(self, limit=50):
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, fault_type, severity, parameter, value, grid_section, alert_sent
                FROM alerts
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            cols = ["timestamp", "fault_type", "severity", "parameter", "value", "grid_section", "alert_sent"]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error("Failed to retrieve alerts: %s", e)
            return []

    def get_alert_statistics(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE DATE(created_at) = DATE('now')")
            today_total = cursor.fetchone()[0]

            cursor.execute(
                "SELECT severity, COUNT(*) FROM alerts WHERE DATE(created_at) = DATE('now') GROUP BY severity"
            )
            severity_breakdown = dict(cursor.fetchall())

            cursor.execute(
                "SELECT COUNT(*) FROM alerts WHERE severity = 'critical' "
                "AND datetime(created_at) > datetime('now', '-1 hour')"
            )
            critical_last_hour = cursor.fetchone()[0]

            return {
                "today_total":       today_total,
                "severity_breakdown": severity_breakdown,
                "critical_last_hour": critical_last_hour,
            }
        except Exception as e:
            self.logger.error("Failed to get alert statistics: %s", e)
            return {"today_total": 0, "severity_breakdown": {}, "critical_last_hour": 0}

    def cleanup_old_alerts(self, days=None):
        if days is None:
            days = self.config["data_settings"]["retention_days"]
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "DELETE FROM alerts WHERE datetime(created_at) < datetime('now', '-{} days')".format(days)
            )
            deleted = cursor.rowcount
            self.conn.commit()
            self.logger.info("Cleaned up %d old alerts", deleted)
            return deleted
        except Exception as e:
            self.logger.error("Failed to cleanup old alerts: %s", e)
            return 0

    def __del__(self):
        if hasattr(self, "conn"):
            self.conn.close()
