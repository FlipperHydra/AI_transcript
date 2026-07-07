"""
db.py — SQLite helpers.

Schema (single-job model):
  jobs        — one row per recording session (id, created_at, status)
  transcripts — one row per job (job_id, content JSON)
  notes       — one row per job (job_id, content markdown)

Notes and transcripts are stored entirely in the DB.
Files are only produced on explicit user export requests.
wav_path dropped: WAV is deleted after processing, storing a dead path is pointless.
"""

import sqlite3
import json
import os
import threading
from pathlib import Path

DB_PATH = Path(os.environ.get("DB_PATH", "/app/output/notes.db"))

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("PRAGMA cache_size=-32000")  # 32MB page cache
    return _conn


def init_db() -> None:
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                created_at  TEXT,
                status      TEXT
            );
            CREATE TABLE IF NOT EXISTS transcripts (
                job_id      TEXT PRIMARY KEY,
                content     TEXT
            );
            CREATE TABLE IF NOT EXISTS notes (
                job_id      TEXT PRIMARY KEY,
                content     TEXT
            );
        """)


# ── Jobs ──────────────────────────────────────────────────────────────────────

def create_job(job_id: str, created_at: str) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO jobs (id, created_at, status) VALUES (?, ?, ?)",
            (job_id, created_at, "processing"),
        )
        conn.commit()


def update_job_status(job_id: str, status: str) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
        conn.commit()


def get_all_jobs() -> list[dict]:
    with _lock:
        rows = _get_conn().execute(
            "SELECT id, created_at, status FROM jobs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# ── Transcripts ───────────────────────────────────────────────────────────────

def save_transcript(job_id: str, segments: list[dict]) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO transcripts (job_id, content) VALUES (?, ?)",
            (job_id, json.dumps(segments, ensure_ascii=False)),
        )
        conn.commit()


def get_transcript(job_id: str) -> list[dict] | None:
    with _lock:
        row = _get_conn().execute(
            "SELECT content FROM transcripts WHERE job_id = ?", (job_id,)
        ).fetchone()
        return json.loads(row["content"]) if row else None


# ── Notes ─────────────────────────────────────────────────────────────────────

def save_notes(job_id: str, content: str) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO notes (job_id, content) VALUES (?, ?)",
            (job_id, content),
        )
        conn.commit()


def get_notes(job_id: str) -> dict | None:
    with _lock:
        row = _get_conn().execute(
            "SELECT job_id, content FROM notes WHERE job_id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def get_all_notes() -> list[dict]:
    """Return metadata only — omit content for list views."""
    with _lock:
        rows = _get_conn().execute(
            """SELECT n.job_id, j.created_at, j.status
               FROM notes n JOIN jobs j ON n.job_id = j.id
               ORDER BY j.created_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
