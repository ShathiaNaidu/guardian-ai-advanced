from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config import APP_TIMEZONE


def local_now() -> datetime:
    """Return the current timezone-aware application time."""
    try:
        return datetime.now(ZoneInfo(APP_TIMEZONE))
    except ZoneInfoNotFoundError:
        # Fall back safely if APP_TIMEZONE was entered incorrectly.
        return datetime.now().astimezone()


def local_today() -> date:
    """Return today's date in the configured application timezone."""
    return local_now().date()


def runtime_datetime_context() -> str:
    """Authoritative date/time context supplied to Gemini."""
    now = local_now()
    return (
        "\n\nVerified runtime date and time supplied by the Guardian AI server:\n"
        f"- Current local ISO date: {now.date().isoformat()}\n"
        f"- Current local weekday: {now.strftime('%A')}\n"
        f"- Current local time: {now.strftime('%H:%M:%S')}\n"
        f"- Configured timezone: {APP_TIMEZONE}\n"
        "Treat these values as authoritative. Never guess today's date or time "
        "from model training, conversation history, or an unverified webpage. "
        "Resolve today, tomorrow, yesterday, this week, next week, this month, "
        "and this year relative to this verified local date."
    )
