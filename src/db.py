"""Shared SQLite connection factory."""
import sqlite3
import os
# Events table statuses
EVENT_STATUS_PENDING = "PENDING"
EVENT_STATUS_COMPLETED = "COMPLETED"
EVENT_STATUS_ERROR = "ERROR"

# Tool intents table statuses
TOOL_INTENT_STATUS_PENDING = "PENDING"
TOOL_INTENT_STATUS_COMPLETED = "COMPLETED"

def create_shared_connection(db_path: str) -> sqlite3.Connection:
    """Return a WAL-mode sqlite3.Connection.

    Args:
        db_path (str): File or `:memory:` for an in-memory database.

    Returns:
        sqlite3.Connection: Shared connection configured for WAL mode,
            normal synchronous, and a busy timeout of 5000ms.
    """
    # LangGraph checkpointers may write checkpoints from worker threads.
    # Allow this shared connection to be used across threads.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.commit()
    return conn

def setup_aer_tables(conn: sqlite3.Connection) -> None:
    """Create tables for the AER database if they don't exist.

    Args:
        conn (sqlite3.Connection): The database connection.
    """ 
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step_id INTEGER NOT NULL,
            step_type TEXT NOT NULL,
            tool_name TEXT,
            input_hash TEXT,
            output TEXT,
            status TEXT NOT NULL DEFAULT '{EVENT_STATUS_PENDING}',
            created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            completed_at TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_events_run_step ON events (run_id, step_id);

        CREATE TABLE IF NOT EXISTS tool_intents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent_hash TEXT NOT NULL UNIQUE,
            tool_name TEXT NOT NULL,
            args_json TEXT,
            status TEXT NOT NULL DEFAULT '{TOOL_INTENT_STATUS_PENDING}',
            result TEXT,
            created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            completed_at TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_tool_intents_hash ON tool_intents (intent_hash, status);
    """)
    conn.commit()

def get_db_size_kb(path: str) -> float:
    """Helper to record SQLite database size in KB."""
    if os.path.exists(path):
        return os.path.getsize(path) / 1024.0
    return 0.0