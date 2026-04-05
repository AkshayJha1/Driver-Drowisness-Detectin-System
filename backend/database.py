"""
SQLite logging for fatigue / drowsiness events.
Keeps a simple local database file — no server required (good for assignments).
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Database file lives next to the project root usage (created on first run)
_DB_PATH = Path(__file__).resolve().parent.parent / "fatigue_logs.db"


def get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the events table if it does not exist.
    Called once when the app starts.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fatigue_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                score REAL NOT NULL,
                blink_rate REAL,
                eye_closed_sec REAL,
                yawn_count INTEGER,
                head_tilt_deg REAL,
                message TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_event(
    level: str,
    score: float,
    blink_rate: float,
    eye_closed_sec: float,
    yawn_count: int,
    head_tilt_deg: float,
    message: str = "",
) -> None:
    """
    Insert one row when fatigue level changes or on a periodic snapshot (optional).
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO fatigue_events
            (timestamp, level, score, blink_rate, eye_closed_sec, yawn_count, head_tilt_deg, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                level,
                float(score),
                float(blink_rate),
                float(eye_closed_sec),
                int(yawn_count),
                float(head_tilt_deg),
                message,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_recent(limit: int = 20) -> List[Dict[str, Any]]:
    """Return the last N log rows (handy for debugging / viva demo)."""
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT * FROM fatigue_events ORDER BY id DESC LIMIT ?;
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
