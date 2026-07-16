from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from config import DB_PATH


def _ensure_parent() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def connection():
    _ensure_parent()
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    schema = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        language TEXT NOT NULL DEFAULT 'English',
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        category TEXT NOT NULL,
        due_date TEXT NOT NULL,
        due_time TEXT,
        notes TEXT,
        completed INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS scam_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        phone_number TEXT NOT NULL,
        report_type TEXT NOT NULL,
        description TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS trusted_contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        phone_number TEXT NOT NULL,
        relationship TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        prompt TEXT NOT NULL,
        response TEXT NOT NULL,
        used_live_search INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT NOT NULL,
        details TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
    );
    """
    with connection() as conn:
        conn.executescript(schema)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def execute(query: str, params: Iterable[Any] = ()) -> int:
    with connection() as conn:
        cur = conn.execute(query, tuple(params))
        return int(cur.lastrowid or 0)


def fetch_one(query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    with connection() as conn:
        row = conn.execute(query, tuple(params)).fetchone()
        return dict(row) if row else None


def fetch_all(query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]


def log_action(user_id: int | None, action: str, details: str = "") -> None:
    execute(
        "INSERT INTO audit_logs(user_id, action, details, created_at) VALUES (?, ?, ?, ?)",
        (user_id, action, details[:2000], now_iso()),
    )
