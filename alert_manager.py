import smtplib
import json
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging

class AlertManager:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.alert_settings = self.config['alert_settings']
        self.setup_database()
        self.setup_logging()
    
    def setup_database(self):
        """Initialize SQLite database for alert logging"""
        self.conn = sqlite3.connect('alerts.db', check_same_thread=False)
        cursor = self.conn.cursor()
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
        self.conn.commit()
    
    def setup_logging(self):
        """Setup logging for alert system"""
        logging.basicConfig(
            filename='alert_system.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def should_send_alert(self, severity):
        """Check if alert should be sent based on severity"""
        return severity in self.alert_settings.get('alert_on_severity', [])
    
    def send_email_alert(self, fault_data):
        """Send email alert for fault"""
        if not self.alert_settings['email_alerts']['enabled']:
            return False
        
        try:
            email_config = self.alert_settings['email_alerts']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"GRID ALERT: {fault_data['fault_type']} - {fault_data['severity'].upper()}"
            
            body = f"""
            SMART GRID FAULT DETECTED
            
            Timestamp: {fault_data['timestamp']}
            Fault Type: {fault_data['fault_type']}
            Severity: {fault_data['severity'].upper()}
            Parameter: {fault_data['parameter']}
            Value: {fault_data['value']}
            Grid Section: {fault_data.get('grid_section', 'Unknown')}
            
            Please investigate immediately if this is a critical fault.
            
            Grid Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent for {fault_data['fault_type']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def log_fault_to_database(self, fault_data):
        """Store fault data in database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (timestamp, fault_type, severity, parameter, value, grid_section, alert_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                fault_data['timestamp'],
                fault_data['fault_type'],
                fault_data['severity'],
                fault_data['parameter'],
                fault_data['value'],
                fault_data.get('grid_section', 'Unknown'),
                fault_data.get('alert_sent', False)
            ))
            self.conn.commit()
            self.logger.info(f"Fault logged to database: {fault_data['fault_type']}")
            
        except Exception as e:
            self.logger.error(f"Failed to log fault to database: {e}")
    
    def process_fault(self, fault_data):
        """Process a detected fault - log and send alerts if needed"""
        # Add severity to fault data
        fault_type = fault_data['fault_type'].lower().replace('-', '_').replace(' ', '_')
        severity = self.get_fault_severity(fault_type)
        fault_data['severity'] = severity
        
        # Log to database
        self.log_fault_to_database(fault_data)
        
        # Send alert if severity warrants it
        if self.should_send_alert(severity):
            alert_sent = self.send_email_alert(fault_data)
            fault_data['alert_sent'] = alert_sent
        
        return fault_data
    
    def get_fault_severity(self, fault_type):
        """Get severity level for fault type"""
        thresholds = self.config['fault_thresholds']
        for threshold_key, threshold_data in thresholds.items():
            if fault_type in threshold_key or threshold_key in fault_type:
                return threshold_data.get('severity', 'info')
        return 'info'
    
    def get_recent_alerts(self, limit=50):
        """Get recent alerts from database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT timestamp, fault_type, severity, parameter, value, grid_section, alert_sent
                FROM alerts 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = ['timestamp', 'fault_type', 'severity', 'parameter', 'value', 'grid_section', 'alert_sent']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve alerts: {e}")
            return []
    
    def get_alert_statistics(self):
        """Get alert statistics for dashboard"""
        try:
            cursor = self.conn.cursor()
            
            # Total alerts today
            cursor.execute('''
                SELECT COUNT(*) FROM alerts 
                WHERE DATE(created_at) = DATE('now')
            ''')
            today_count = cursor.fetchone()[0]
            
            # Alerts by severity
            cursor.execute('''
                SELECT severity, COUNT(*) FROM alerts 
                WHERE DATE(created_at) = DATE('now')
                GROUP BY severity
            ''')
            severity_counts = dict(cursor.fetchall())
            
            # Critical alerts in last hour
            cursor.execute('''
                SELECT COUNT(*) FROM alerts 
                WHERE severity = 'critical' 
                AND datetime(created_at) > datetime('now', '-1 hour')
            ''')
            critical_last_hour = cursor.fetchone()[0]
            
            return {
                'today_total': today_count,
                'severity_breakdown': severity_counts,
                'critical_last_hour': critical_last_hour
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get alert statistics: {e}")
            return {'today_total': 0, 'severity_breakdown': {}, 'critical_last_hour': 0}
    
    def cleanup_old_alerts(self, days=None):
        """Clean up old alerts based on retention policy"""
        if days is None:
            days = self.config['data_settings']['retention_days']
        
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                DELETE FROM alerts 
                WHERE datetime(created_at) < datetime('now', '-{} days')
            '''.format(days))
            
            deleted_count = cursor.rowcount
            self.conn.commit()
            self.logger.info(f"Cleaned up {deleted_count} old alerts")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old alerts: {e}")
            return 0
    
    def __del__(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()