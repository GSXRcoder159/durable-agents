"""Shared SQLite connection factory."""
import sqlite3

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
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step_id INTEGER NOT NULL,
            step_type TEXT NOT NULL,
            tool_name TEXT,
            input_hash TEXT,
            output TEXT,
            status TEXT NOT NULL DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            completed_at TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_events_run_step ON events (run_id, step_id);
    """)
    conn.commit()
