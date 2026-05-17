import sqlite3
import json
import uuid
import datetime
from pathlib import Path

class SessionLog:
    """
    Records user interactions to a structured log that supports future user studies.
    This thesis does not conduct a user study; the log infrastructure exists as 
    future-work scaffolding.
    """
    def __init__(self, db_path="outputs/sessions/session_log.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP NULL,
                    input_mode TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    created_at TIMESTAMP,
                    source_type TEXT,
                    source_filename TEXT NULL,
                    total_stories INTEGER,
                    stories_reviewed INTEGER DEFAULT 0,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stories (
                    story_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    batch_id TEXT NULL,
                    queue_position INTEGER NULL,
                    submitted_at TIMESTAMP,
                    original_text TEXT,
                    pipeline_outputs_json TEXT,
                    review_status TEXT DEFAULT 'pending',
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY(batch_id) REFERENCES batches(batch_id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    batch_id TEXT NULL,
                    story_id TEXT NULL,
                    timestamp TIMESTAMP,
                    event_type TEXT,
                    payload_json TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            ''')
            conn.commit()

    def start_session(self, input_mode):
        session_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO sessions (session_id, started_at, input_mode)
                VALUES (?, ?, ?)
            ''', (session_id, datetime.datetime.now(), input_mode))
        return session_id

    def end_session(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE sessions SET ended_at = ? WHERE session_id = ?
            ''', (datetime.datetime.now(), session_id))

    def start_batch(self, session_id, source_type, source_filename, total_stories):
        batch_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO batches (batch_id, session_id, created_at, source_type, source_filename, total_stories)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (batch_id, session_id, datetime.datetime.now(), source_type, source_filename, total_stories))
        return batch_id

    def log_event(self, session_id, event_type, payload, story_id=None, batch_id=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO events (session_id, batch_id, story_id, timestamp, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, batch_id, story_id, datetime.datetime.now(), event_type, json.dumps(payload)))

    def log_story(self, session_id, story_id, original_text, pipeline_outputs, batch_id=None, queue_position=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO stories (story_id, session_id, batch_id, queue_position, submitted_at, original_text, pipeline_outputs_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (story_id, session_id, batch_id, queue_position, datetime.datetime.now(), original_text, json.dumps(pipeline_outputs)))

    def update_story_status(self, story_id, status):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE stories SET review_status = ? WHERE story_id = ?
            ''', (status, story_id))
            if status in ['accepted', 'skipped']:
                cursor = conn.execute('SELECT batch_id FROM stories WHERE story_id = ?', (story_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    conn.execute('UPDATE batches SET stories_reviewed = stories_reviewed + 1 WHERE batch_id = ?', (row[0],))

    def get_session_data(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            events = [dict(row) for row in conn.execute('SELECT * FROM events WHERE session_id = ? ORDER BY timestamp ASC', (session_id,))]
            stories = [dict(row) for row in conn.execute('SELECT * FROM stories WHERE session_id = ? ORDER BY submitted_at ASC', (session_id,))]
            batches = [dict(row) for row in conn.execute('SELECT * FROM batches WHERE session_id = ? ORDER BY created_at ASC', (session_id,))]
        return {"events": events, "stories": stories, "batches": batches}

    def get_session_summary(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM stories WHERE session_id = ?', (session_id,))
            stories_processed = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM stories WHERE session_id = ? AND review_status = "accepted"', (session_id,))
            accepted = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM stories WHERE session_id = ? AND review_status = "skipped"', (session_id,))
            skipped = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM events WHERE session_id = ? AND event_type = "REFINEMENT_REGENERATED"', (session_id,))
            regenerated = cursor.fetchone()[0]
            
            cursor.execute('SELECT input_mode, started_at FROM sessions WHERE session_id = ?', (session_id,))
            row = cursor.fetchone()
            input_mode = row[0] if row else "unknown"
            started_at = row[1] if row else None

        return {
            "stories_processed": stories_processed,
            "accepted": accepted,
            "regenerated": regenerated,
            "skipped": skipped,
            "input_mode": input_mode,
            "started_at": started_at
        }

    def get_batch_progress(self, batch_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT total_stories, stories_reviewed FROM batches WHERE batch_id = ?', (batch_id,))
            row = cursor.fetchone()
            if not row:
                return {}
            total_stories, stories_reviewed = row
            
            cursor.execute('SELECT COUNT(*) FROM stories WHERE batch_id = ? AND review_status = "accepted"', (batch_id,))
            accepted = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM events WHERE batch_id = ? AND event_type = "REFINEMENT_REGENERATED"', (batch_id,))
            regenerated = cursor.fetchone()[0]
            
        return {
            "total_stories": total_stories,
            "stories_reviewed": stories_reviewed,
            "accepted": accepted,
            "regenerated": regenerated
        }
