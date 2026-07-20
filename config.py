from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _setting(name: str, default: str = "") -> str:
    """Read Streamlit secrets first, then environment variables."""
    value: Any = ""
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
    except Exception:
        value = ""

    if value is None or str(value).strip() == "":
        value = os.getenv(name, default)

    return str(value).strip()


APP_NAME = "Guardian AI Advanced"
APP_COUNTRY = _setting("APP_COUNTRY", "Malaysia")
APP_TIMEZONE = _setting("APP_TIMEZONE", "Asia/Kuala_Lumpur")
DB_PATH = Path(_setting("GUARDIAN_DB_PATH", str(BASE_DIR / "data" / "guardian.db")))

# Gemini handles chat, grounded Google Search, image understanding, and
# voice-note transcription. Use current stable model names in Streamlit Secrets.
GEMINI_API_KEY = _setting("GEMINI_API_KEY")
GEMINI_MODEL = _setting("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_SEARCH_MODEL = _setting("GEMINI_SEARCH_MODEL", GEMINI_MODEL)
GEMINI_VISION_MODEL = _setting("GEMINI_VISION_MODEL", GEMINI_MODEL)
GEMINI_AUDIO_MODEL = _setting("GEMINI_AUDIO_MODEL", GEMINI_MODEL)
GEMINI_FALLBACK_MODELS = tuple(
    model.strip()
    for model in _setting(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.1-flash-lite,gemini-2.5-flash-lite,gemini-3.5-flash",
    ).split(",")
    if model.strip()
)

# Azure Speech is used only for natural spoken AI replies.
AZURE_SPEECH_KEY = _setting("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = _setting("AZURE_SPEECH_REGION")
AZURE_SPEECH_ENDPOINT = _setting("AZURE_SPEECH_ENDPOINT")

NPRA_PRODUCT_SEARCH_URL = "https://quest3plus.bpfk.gov.my/pmo2/index.php"
MALAYSIA_EMERGENCY_NUMBER = "999"
