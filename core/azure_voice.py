from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.sax.saxutils import escape

import streamlit as st

from config import AZURE_SPEECH_ENDPOINT, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION


VOICE_PROFILES: dict[str, dict[str, str]] = {
    "English": {
        "locale": "en-US",
        "female": "en-US-JennyNeural",
        "male": "en-US-GuyNeural",
    },
    "Bahasa Melayu": {
        "locale": "ms-MY",
        "female": "ms-MY-YasminNeural",
        "male": "ms-MY-OsmanNeural",
    },
    "Malay": {
        "locale": "ms-MY",
        "female": "ms-MY-YasminNeural",
        "male": "ms-MY-OsmanNeural",
    },
    "Tamil": {
        "locale": "ta-MY",
        "female": "ta-MY-KaniNeural",
        "male": "ta-MY-SuryaNeural",
    },
    "Chinese": {
        "locale": "zh-CN",
        "female": "zh-CN-XiaoxiaoNeural",
        "male": "zh-CN-YunxiNeural",
    },
    "Hindi": {
        "locale": "hi-IN",
        "female": "hi-IN-SwaraNeural",
        "male": "hi-IN-MadhurNeural",
    },
    "Indonesian": {
        "locale": "id-ID",
        "female": "id-ID-GadisNeural",
        "male": "id-ID-ArdiNeural",
    },
    "Arabic": {
        "locale": "ar-SA",
        "female": "ar-SA-ZariyahNeural",
        "male": "ar-SA-HamedNeural",
    },
    "Spanish": {
        "locale": "es-ES",
        "female": "es-ES-ElviraNeural",
        "male": "es-ES-AlvaroNeural",
    },
    "French": {
        "locale": "fr-FR",
        "female": "fr-FR-DeniseNeural",
        "male": "fr-FR-HenriNeural",
    },
    "Japanese": {
        "locale": "ja-JP",
        "female": "ja-JP-NanamiNeural",
        "male": "ja-JP-KeitaNeural",
    },
    "Korean": {
        "locale": "ko-KR",
        "female": "ko-KR-SunHiNeural",
        "male": "ko-KR-InJoonNeural",
    },
}


def _normalise_region(region: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (region or "").lower())


def configured_region() -> str:
    return _normalise_region(AZURE_SPEECH_REGION)


def is_configured() -> bool:
    return bool(AZURE_SPEECH_KEY and (configured_region() or AZURE_SPEECH_ENDPOINT))


def configuration_error() -> str:
    missing: list[str] = []
    if not AZURE_SPEECH_KEY:
        missing.append("AZURE_SPEECH_KEY")
    if not configured_region() and not AZURE_SPEECH_ENDPOINT:
        missing.append("AZURE_SPEECH_REGION")

    if not missing:
        return ""

    return (
        "Azure Neural voice is not configured. Add "
        + " and ".join(f"`{name}`" for name in missing)
        + " in Streamlit App settings → Secrets, save, and reboot the app."
    )


def _profile(language: str) -> dict[str, str]:
    return VOICE_PROFILES.get(language, VOICE_PROFILES["English"])


def _gender_key(voice_preference: str) -> str:
    return "male" if "male" in (voice_preference or "").lower() else "female"


def voice_for_language(
    language: str,
    voice_preference: str = "Natural female voice",
) -> str:
    return _profile(language)[_gender_key(voice_preference)]


def locale_for_language(language: str) -> str:
    return _profile(language)["locale"]


def _base_endpoint() -> str:
    if AZURE_SPEECH_ENDPOINT:
        return AZURE_SPEECH_ENDPOINT.rstrip("/")

    region = configured_region()
    if not region:
        raise RuntimeError(configuration_error())

    return f"https://{region}.tts.speech.microsoft.com"


def _synthesis_endpoint() -> str:
    return f"{_base_endpoint()}/cognitiveservices/v1"


def _voices_endpoint() -> str:
    return f"{_base_endpoint()}/cognitiveservices/voices/list"


def _clean_text(text: str, maximum_characters: int = 5000) -> str:
    clean = " ".join((text or "").split())
    if not clean:
        raise ValueError("There is no reply text to speak.")

    if len(clean) > maximum_characters:
        clean = clean[:maximum_characters].rsplit(" ", 1)[0]
        clean += ". The remaining reply is available on screen."

    return clean


def _build_ssml(
    text: str,
    language: str,
    voice_preference: str,
    voice_name: str | None = None,
) -> bytes:
    locale = locale_for_language(language)
    voice = voice_name or voice_for_language(language, voice_preference)
    safe_text = escape(_clean_text(text))

    document = f"""<speak version="1.0"
        xmlns="http://www.w3.org/2001/10/synthesis"
        xml:lang="{locale}">
      <voice name="{voice}">
        <prosody rate="+0%" pitch="+0Hz" volume="+0%">
          {safe_text}
        </prosody>
      </voice>
    </speak>"""

    return document.encode("utf-8")


def _http_error_message(exc: HTTPError) -> str:
    detail = exc.read().decode("utf-8", errors="replace").strip()
    return f"Azure Speech HTTP {exc.code}: {detail or exc.reason}"


def _request_audio(
    text: str,
    language: str,
    voice_preference: str,
    voice_name: str | None = None,
) -> bytes:
    if not is_configured():
        raise RuntimeError(configuration_error())

    request = Request(
        _synthesis_endpoint(),
        data=_build_ssml(text, language, voice_preference, voice_name),
        method="POST",
        headers={
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-96kbitrate-mono-mp3",
            "User-Agent": "Guardian-AI-Advanced",
        },
    )

    try:
        with urlopen(request, timeout=45) as response:
            audio = response.read()
    except HTTPError as exc:
        raise RuntimeError(_http_error_message(exc)) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Azure Speech: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            "Azure Speech did not respond before the request timed out."
        ) from exc

    if len(audio) < 200:
        raise RuntimeError("Azure returned an empty or incomplete audio file.")

    return audio


def _list_voices() -> list[dict[str, Any]]:
    if not is_configured():
        raise RuntimeError(configuration_error())

    request = Request(
        _voices_endpoint(),
        method="GET",
        headers={
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "User-Agent": "Guardian-AI-Advanced",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(_http_error_message(exc)) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Azure Speech: {exc.reason}") from exc

    parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise RuntimeError("Azure returned an unexpected voice-list response.")

    return parsed


def _regional_voice_fallback(
    language: str,
    voice_preference: str,
) -> str | None:
    locale = locale_for_language(language).lower()
    gender = "male" if _gender_key(voice_preference) == "male" else "female"

    voices = _list_voices()
    for item in voices:
        if not isinstance(item, dict):
            continue
        item_locale = str(item.get("Locale", "")).lower()
        item_gender = str(item.get("Gender", "")).lower()
        short_name = str(item.get("ShortName", "")).strip()
        if item_locale == locale and item_gender == gender and short_name:
            return short_name

    for item in voices:
        if not isinstance(item, dict):
            continue
        item_locale = str(item.get("Locale", "")).lower()
        short_name = str(item.get("ShortName", "")).strip()
        if item_locale == locale and short_name:
            return short_name

    return None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=100)
def synthesize_mp3(
    text: str,
    language: str = "English",
    voice_preference: str = "Natural female voice",
) -> bytes:
    """
    Generate natural speech as MP3 using Azure Neural TTS.

    Identical speech requests are cached for one hour to avoid repeatedly
    consuming the Azure character allowance.
    """
    try:
        return _request_audio(text, language, voice_preference)
    except RuntimeError as exc:
        # A requested voice can occasionally be unavailable in a particular
        # region. Resolve another voice with the same locale and gender.
        if "HTTP 400" not in str(exc):
            raise

        fallback_voice = _regional_voice_fallback(language, voice_preference)
        expected_voice = voice_for_language(language, voice_preference)
        if not fallback_voice or fallback_voice == expected_voice:
            raise

        return _request_audio(
            text,
            language,
            voice_preference,
            voice_name=fallback_voice,
        )


def test_connection(
    language: str = "English",
    voice_preference: str = "Natural female voice",
) -> tuple[bool, str]:
    try:
        expected_voice = voice_for_language(language, voice_preference)
        voices = _list_voices()
        names = {
            str(item.get("ShortName", ""))
            for item in voices
            if isinstance(item, dict)
        }

        if expected_voice in names:
            return (
                True,
                f"Azure Neural TTS connected successfully. "
                f"Selected voice: `{expected_voice}`.",
            )

        fallback = _regional_voice_fallback(language, voice_preference)
        if fallback:
            return (
                True,
                f"Azure connected successfully. `{expected_voice}` is not "
                f"listed in this region, so Guardian AI will use `{fallback}`.",
            )

        return (
            False,
            f"Azure connected, but no suitable {language} voice was found "
            f"in region `{configured_region()}`.",
        )
    except Exception as exc:
        return False, friendly_error(exc)


def friendly_error(exc: Exception) -> str:
    message = str(exc).strip()
    lower = message.lower()

    if "http 401" in lower or "unauthorized" in lower:
        return (
            "Azure rejected the credentials. Confirm that AZURE_SPEECH_KEY "
            "and AZURE_SPEECH_REGION belong to the same Speech resource."
        )

    if "http 403" in lower or "forbidden" in lower:
        return (
            "Azure denied access. Confirm that the Speech resource is active "
            "and its networking setting permits public access."
        )

    if "http 429" in lower or "too many requests" in lower:
        return (
            "The Azure Speech quota or request rate was reached. The free F0 "
            "tier is not unlimited; wait for the allowance to reset or move "
            "to a paid Speech tier."
        )

    if "http 400" in lower or "bad request" in lower:
        return (
            "Azure rejected the speech request. Try the connection test and "
            "confirm that a matching voice is available in the resource region."
        )

    if "not configured" in lower:
        return configuration_error()

    return message or "An unknown Azure Neural voice error occurred."
