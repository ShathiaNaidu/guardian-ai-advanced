from __future__ import annotations

from core.database import execute, fetch_all, now_iso


def add_reminder(user_id: int, title: str, category: str, due_date: str, due_time: str = "", notes: str = "") -> int:
    return execute(
        "INSERT INTO reminders(user_id, title, category, due_date, due_time, notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, title.strip(), category, due_date, due_time, notes.strip()[:2000], now_iso()),
    )


def list_reminders(user_id: int, include_completed: bool = True) -> list[dict]:
    query = "SELECT * FROM reminders WHERE user_id = ?"
    params: list = [user_id]
    if not include_completed:
        query += " AND completed = 0"
    query += " ORDER BY completed ASC, due_date ASC, COALESCE(due_time, '') ASC"
    return fetch_all(query, params)


def set_completed(reminder_id: int, user_id: int, completed: bool) -> None:
    execute("UPDATE reminders SET completed = ? WHERE id = ? AND user_id = ?", (1 if completed else 0, reminder_id, user_id))


def delete_reminder(reminder_id: int, user_id: int) -> None:
    execute("DELETE FROM reminders WHERE id = ? AND user_id = ?", (reminder_id, user_id))
