"""Structured audit log backed by SQLite.

One row per event. event_type is "decision" (a classification) or "appeal".
Columns cover what every milestone needs; M3 fills the decision fields, M4 adds
the second signal score, M5 adds appeals. `detail` is a JSON blob for the
free-form bits (rationale, stylometry features, label text, appeal reasoning)
so the schema does not churn every milestone.
"""

import json
import sqlite3
from datetime import datetime, timezone

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type           TEXT    NOT NULL,
    content_id           TEXT    NOT NULL,
    creator_id           TEXT,
    timestamp            TEXT    NOT NULL,
    attribution          TEXT,
    p_ai                 REAL,
    confidence           REAL,
    llm_score            REAL,
    stylometry_score     REAL,
    status               TEXT,
    label_variant        TEXT,
    detail               TEXT
);
"""


def _connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the events table if it does not exist. Safe to call repeatedly."""
    with _connect() as conn:
        conn.execute(_SCHEMA)


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def log_event(
    event_type,
    content_id,
    creator_id=None,
    attribution=None,
    p_ai=None,
    confidence=None,
    llm_score=None,
    stylometry_score=None,
    status=None,
    label_variant=None,
    detail=None,
):
    """Append one event row and return it as a dict (with its assigned id)."""
    timestamp = _utc_now_iso()
    detail_json = json.dumps(detail) if detail is not None else None
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO events (
                event_type, content_id, creator_id, timestamp, attribution,
                p_ai, confidence, llm_score, stylometry_score, status,
                label_variant, detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_type, content_id, creator_id, timestamp, attribution,
                p_ai, confidence, llm_score, stylometry_score, status,
                label_variant, detail_json,
            ),
        )
        event_id = cursor.lastrowid
    return {
        "id": event_id,
        "event_type": event_type,
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": timestamp,
        "attribution": attribution,
        "p_ai": p_ai,
        "confidence": confidence,
        "llm_score": llm_score,
        "stylometry_score": stylometry_score,
        "status": status,
        "label_variant": label_variant,
        "detail": detail,
    }


def update_status(content_id, status):
    """Set the status on every decision row for a content_id (used when an appeal
    flips the content to 'under_review'). Returns the number of rows updated."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE events SET status = ? WHERE content_id = ? AND event_type = 'decision'",
            (status, content_id),
        )
        return cursor.rowcount


def _row_to_dict(row):
    entry = dict(row)
    if entry.get("detail"):
        try:
            entry["detail"] = json.loads(entry["detail"])
        except (TypeError, ValueError):
            pass
    return entry


def get_log(limit=50):
    """Return the most recent events first, as a list of dicts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_latest_decision(content_id):
    """Return the most recent 'decision' event for a content_id, or None.

    Used by the appeal endpoint (M5) to confirm a content_id is real and to
    show the reviewer the original decision an appeal contests.
    """
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM events
            WHERE content_id = ? AND event_type = 'decision'
            ORDER BY id DESC LIMIT 1
            """,
            (content_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None
