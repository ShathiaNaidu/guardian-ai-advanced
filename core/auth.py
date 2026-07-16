from __future__ import annotations

import hashlib
import hmac
import re
import secrets

from core.database import execute, fetch_one, now_iso

PBKDF2_ITERATIONS = 240_000


def _hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return digest.hex()


def validate_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return False, "Password must include at least one letter and one number."
    return True, ""


def register_user(username: str, full_name: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    full_name = full_name.strip()
    if not re.fullmatch(r"[a-z0-9_.-]{3,30}", username):
        return False, "Username must be 3–30 characters using letters, numbers, dot, dash or underscore."
    if len(full_name) < 2:
        return False, "Please enter your full name."
    valid, message = validate_password(password)
    if not valid:
        return False, message
    if fetch_one("SELECT id FROM users WHERE username = ?", (username,)):
        return False, "That username already exists."

    first_user = fetch_one("SELECT COUNT(*) AS count FROM users")
    role = "admin" if not first_user or first_user["count"] == 0 else "user"
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    execute(
        "INSERT INTO users(username, full_name, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (username, full_name, password_hash, salt, role, now_iso()),
    )
    role_note = " You are the first user, so your account is an administrator." if role == "admin" else ""
    return True, f"Account created successfully.{role_note}"


def authenticate(username: str, password: str) -> dict | None:
    user = fetch_one("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
    if not user:
        return None
    candidate = _hash_password(password, user["salt"])
    return user if hmac.compare_digest(candidate, user["password_hash"]) else None


def update_profile(user_id: int, full_name: str, language: str) -> None:
    execute("UPDATE users SET full_name = ?, language = ? WHERE id = ?", (full_name.strip(), language, user_id))


def change_password(user_id: int, old_password: str, new_password: str) -> tuple[bool, str]:
    user = fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not user or not hmac.compare_digest(_hash_password(old_password, user["salt"]), user["password_hash"]):
        return False, "Current password is incorrect."
    valid, message = validate_password(new_password)
    if not valid:
        return False, message
    salt = secrets.token_hex(16)
    execute("UPDATE users SET password_hash = ?, salt = ? WHERE id = ?", (_hash_password(new_password, salt), salt, user_id))
    return True, "Password updated."
