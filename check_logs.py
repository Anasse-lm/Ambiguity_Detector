import sqlite3
import sys
import os

db_path = "outputs/sessions/session_log.db"
if not os.path.exists(db_path):
    print("No DB found at", db_path)
    sys.exit(0)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT event_type, payload_json, timestamp FROM events ORDER BY timestamp DESC LIMIT 20")
    for row in cursor.fetchall():
        if "ERROR" in row[0]:
            print("ERROR found:", row)
except Exception as e:
    print(e)
