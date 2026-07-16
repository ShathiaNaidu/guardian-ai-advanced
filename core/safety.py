from __future__ import annotations

import ipaddress
import re
from collections import Counter
from urllib.parse import urlparse

import cv2
import numpy as np
import phonenumbers

from core.database import execute, fetch_all, now_iso

SCAM_RULES: list[tuple[str, int, str]] = [
    (r"\b(otp|one[- ]time password|tac|pin|password)\b", 18, "Requests confidential authentication information"),
    (r"\b(urgent|immediately|within \d+ (minutes|hours)|final warning)\b", 10, "Creates urgency or fear"),
    (r"\b(account (will be )?(blocked|suspended|frozen)|legal action|police case)\b", 14, "Threatens account or legal consequences"),
    (r"\b(won|winner|prize|lottery|free gift|reward)\b", 12, "Unexpected prize or reward"),
    (r"\b(transfer|bank in|send money|payment|deposit|crypto|bitcoin|gift card)\b", 10, "Requests payment or transfer"),
    (r"\b(click|open|verify|update)\b.{0,45}\b(link|account|details)\b", 12, "Pushes user to open a link or verify details"),
    (r"\b(bank negara|maybank|cimb|public bank|police|lhdn|courier|customs)\b", 5, "Uses an authority or trusted organisation name"),
    (r"\b(job offer|part[- ]time job|easy income|commission|task job)\b", 8, "Possible job or task scam language"),
    (r"\b(investment|guaranteed return|double your money|high return)\b", 14, "Possible investment scam language"),
]

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rebrand.ly", "shorturl.at"}
SUSPICIOUS_TLDS = {"zip", "mov", "top", "click", "work", "support", "country", "gq", "tk"}
BRAND_WORDS = {"maybank", "cimb", "paypal", "apple", "google", "microsoft", "bank", "netflix", "whatsapp"}


def analyze_scam_text(text: str) -> dict:
    lowered = text.lower().strip()
    score = 0
    indicators: list[str] = []
    for pattern, points, explanation in SCAM_RULES:
        if re.search(pattern, lowered, flags=re.I | re.S):
            score += points
            indicators.append(explanation)
    if re.search(r"https?://|www\.", lowered):
        score += 6
        indicators.append("Contains a website link")
    if len(re.findall(r"[!?]", text)) >= 4:
        score += 5
        indicators.append("Uses excessive urgency punctuation")
    if re.search(r"\b\d{10,16}\b", lowered):
        score += 5
        indicators.append("Contains a long account or reference number")
    # Multiple social-engineering signals together are more dangerous than any one phrase.
    if len(indicators) >= 4:
        score += 15
    elif len(indicators) >= 2:
        score += 6
    score = min(score, 100)
    if score >= 65:
        level = "High risk"
    elif score >= 35:
        level = "Suspicious"
    else:
        level = "Low detected risk"
    return {
        "score": score,
        "level": level,
        "indicators": list(dict.fromkeys(indicators)),
        "advice": "Do not share OTPs, passwords or banking PINs. Verify using an official phone number or website you locate independently.",
    }


def analyze_url(raw_url: str) -> dict:
    raw_url = raw_url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_url):
        raw_url = "https://" + raw_url
    parsed = urlparse(raw_url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    score = 0
    indicators: list[str] = []

    if parsed.scheme != "https":
        score += 15
        indicators.append("Does not use HTTPS")
    if hostname in SHORTENERS:
        score += 20
        indicators.append("Uses a shortened link that hides the destination")
    try:
        ipaddress.ip_address(hostname)
        score += 25
        indicators.append("Uses a numeric IP address instead of a normal domain")
    except ValueError:
        pass
    if "xn--" in hostname:
        score += 20
        indicators.append("Contains an internationalized/punycode domain")
    tld = hostname.rsplit(".", 1)[-1] if "." in hostname else ""
    if tld in SUSPICIOUS_TLDS:
        score += 10
        indicators.append(f"Uses a commonly abused .{tld} domain")
    if hostname.count("-") >= 3 or hostname.count(".") >= 4:
        score += 8
        indicators.append("Domain structure is unusually complex")
    if "@" in parsed.netloc:
        score += 20
        indicators.append("Contains an @ symbol that may hide the real destination")
    if len(raw_url) > 120:
        score += 8
        indicators.append("URL is unusually long")
    brand_hits = [word for word in BRAND_WORDS if word in hostname]
    if brand_hits and not any(hostname == b + ".com" or hostname.endswith("." + b + ".com") for b in brand_hits):
        score += 8
        indicators.append("Uses a brand-like word in an unverified domain")
    suspicious_path_words = re.findall(r"login|verify|secure|update|wallet|claim|reward|otp", (parsed.path + " " + parsed.query).lower())
    if suspicious_path_words:
        score += min(15, 4 * len(set(suspicious_path_words)))
        indicators.append("Link path contains account-verification or reward wording")

    score = min(score, 100)
    level = "High risk" if score >= 60 else "Suspicious" if score >= 30 else "Low detected risk"
    return {
        "normalized_url": raw_url,
        "hostname": hostname,
        "score": score,
        "level": level,
        "indicators": list(dict.fromkeys(indicators)),
        "warning": "This is a technical risk check, not proof that a website is safe or malicious. Do not sign in or pay unless independently verified.",
    }


def decode_qr(image_bytes: bytes) -> list[str]:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return []
    detector = cv2.QRCodeDetector()
    values: list[str] = []
    try:
        ok, decoded_info, _points, _ = detector.detectAndDecodeMulti(image)
        if ok:
            values.extend([value for value in decoded_info if value])
    except Exception:
        pass
    if not values:
        value, _points, _ = detector.detectAndDecode(image)
        if value:
            values.append(value)
    return list(dict.fromkeys(values))


def normalize_phone(number: str, default_region: str = "MY") -> str:
    number = number.strip()
    try:
        parsed = phonenumbers.parse(number, default_region)
        if phonenumbers.is_possible_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        pass
    return re.sub(r"[^0-9+]", "", number)


def add_phone_report(user_id: int, number: str, report_type: str, description: str) -> int:
    normalized = normalize_phone(number)
    return execute(
        "INSERT INTO scam_reports(user_id, phone_number, report_type, description, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, normalized, report_type, description.strip()[:2000], now_iso()),
    )


def phone_reputation(number: str) -> dict:
    normalized = normalize_phone(number)
    reports = fetch_all("SELECT report_type, description, created_at FROM scam_reports WHERE phone_number = ? ORDER BY id DESC", (normalized,))
    counts = Counter(report["report_type"] for report in reports)
    score = min(100, len(reports) * 15 + len(counts) * 5)
    level = "High community risk" if score >= 60 else "Reported" if reports else "No local reports"
    return {
        "phone_number": normalized,
        "report_count": len(reports),
        "types": dict(counts),
        "score": score,
        "level": level,
        "reports": reports[:20],
        "note": "No report does not mean a number is safe. This database only contains reports submitted to this Guardian AI installation.",
    }
