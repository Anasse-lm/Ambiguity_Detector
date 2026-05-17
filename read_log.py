import sqlite3
import sys

conn = sqlite3.connect('outputs/session_logs.db')
cursor = conn.cursor()
cursor.execute("SELECT event_type, event_data FROM events WHERE event_type = 'ERROR' ORDER BY timestamp DESC LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(row)
