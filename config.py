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

OPENAI_API_KEY = _setting("OPENAI_API_KEY")
OPENAI_MODEL = _setting("OPENAI_MODEL", "gpt-5-mini")
OPENAI_VISION_MODEL = _setting("OPENAI_VISION_MODEL", OPENAI_MODEL)
OPENAI_TRANSCRIBE_MODEL = _setting("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")

AZURE_SPEECH_KEY = _setting("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = _setting("AZURE_SPEECH_REGION")
AZURE_SPEECH_ENDPOINT = _setting("AZURE_SPEECH_ENDPOINT")

NPRA_PRODUCT_SEARCH_URL = "https://quest3plus.bpfk.gov.my/pmo2/index.php"
MALAYSIA_EMERGENCY_NUMBER = "999"
