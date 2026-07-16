from __future__ import annotations

import calendar
import json
import re
from datetime import date, datetime
from typing import Any

from core.ai_assistant import analyze_image_json

MAL_PATTERN = re.compile(r"\bMAL\s*\d{8,12}\s*[A-Z]?\b", re.I)


def extract_mal_numbers(text: str) -> list[str]:
    return [re.sub(r"\s+", "", item.upper()) for item in MAL_PATTERN.findall(text or "")]


def _safe_date(year: int, month: int, day: int | None = None) -> date | None:
    try:
        if year < 100:
            year += 2000
        if not day or day == 0:
            day = calendar.monthrange(year, month)[1]
        return date(year, month, day)
    except Exception:
        return None


def parse_expiry_dates(text: str) -> list[dict[str, Any]]:
    text = (text or "").upper()
    results: list[dict[str, Any]] = []
    patterns = [
        (r"(?:EXP|EXPIRY|USE BEFORE|BEST BEFORE|BB)\s*[:\-]?\s*(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})", "DMY"),
        (r"(?:EXP|EXPIRY|USE BEFORE|BEST BEFORE|BB)\s*[:\-]?\s*(\d{1,2})[./\-](\d{2,4})", "MY"),
        (r"(?:EXP|EXPIRY|USE BEFORE|BEST BEFORE|BB)\s*[:\-]?\s*(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})", "YMD"),
    ]
    for pattern, mode in patterns:
        for match in re.finditer(pattern, text):
            groups = [int(value) for value in match.groups()]
            parsed: date | None = None
            if mode == "DMY":
                parsed = _safe_date(groups[2], groups[1], groups[0])
            elif mode == "MY":
                parsed = _safe_date(groups[1], groups[0], None)
            elif mode == "YMD":
                parsed = _safe_date(groups[0], groups[1], groups[2])
            if parsed:
                results.append({"date": parsed, "source": match.group(0)})
    unique: dict[str, dict[str, Any]] = {}
    for item in results:
        unique[item["date"].isoformat()] = item
    return list(unique.values())


def parse_gs1(data: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", data or "")
    result: dict[str, Any] = {"raw": data}
    parenthesized = dict(re.findall(r"\((01|10|17|21)\)([^()]+)", compact))
    if parenthesized:
        if "01" in parenthesized:
            result["gtin"] = parenthesized["01"][:14]
        if "17" in parenthesized:
            expiry_raw = parenthesized["17"][:6]
            if re.fullmatch(r"\d{6}", expiry_raw):
                parsed = _safe_date(int(expiry_raw[:2]), int(expiry_raw[2:4]), int(expiry_raw[4:6]))
                result["expiry_date"] = parsed.isoformat() if parsed else None
        if "10" in parenthesized:
            result["batch"] = parenthesized["10"]
        if "21" in parenthesized:
            result["serial"] = parenthesized["21"]
        return result

    match_01 = re.search(r"01(\d{14})", compact)
    match_17 = re.search(r"17(\d{6})", compact)
    if match_01:
        result["gtin"] = match_01.group(1)
    if match_17:
        raw = match_17.group(1)
        parsed = _safe_date(int(raw[:2]), int(raw[2:4]), int(raw[4:6]))
        result["expiry_date"] = parsed.isoformat() if parsed else None
    return result


def expiry_status(expiry: str | date | None) -> dict[str, Any]:
    if not expiry:
        return {"status": "Unverified", "days_remaining": None, "message": "No reliable expiry date was found."}
    expiry_date = date.fromisoformat(expiry) if isinstance(expiry, str) else expiry
    days = (expiry_date - date.today()).days
    if days < 0:
        return {"status": "Expired", "days_remaining": days, "message": "The detected expiry date has passed. Do not use it."}
    if days <= 30:
        return {"status": "Expiring soon", "days_remaining": days, "message": "The detected date is within 30 days."}
    return {"status": "Not expired by detected date", "days_remaining": days, "message": "This does not confirm authenticity, correct storage or suitability for the user."}


def inspect_medicine_image(image_bytes: bytes) -> dict[str, Any]:
    prompt = """
Inspect this medicine package image carefully. Extract only visible information; never guess.
Return JSON with these keys:
product_name, strength, dosage_form, manufacturer, mal_number, expiry_date,
batch_number, qr_or_data_matrix_text, warnings, unreadable_fields.
Use null for information not clearly visible. expiry_date must be YYYY-MM-DD when a full date is visible,
or YYYY-MM when only month/year is visible. State that printed text may require manual confirmation.
"""
    return analyze_image_json(image_bytes, prompt)


def parse_manual_medicine_text(text: str) -> dict[str, Any]:
    expiries = parse_expiry_dates(text)
    expiry = expiries[0]["date"].isoformat() if expiries else None
    return {
        "mal_numbers": extract_mal_numbers(text),
        "expiry_candidates": [{"date": item["date"].isoformat(), "source": item["source"]} for item in expiries],
        "expiry_status": expiry_status(expiry),
    }
