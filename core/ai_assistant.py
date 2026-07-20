from __future__ import annotations

import io
import json
import mimetypes
from typing import Any

from PIL import Image

from config import (
    APP_COUNTRY,
    APP_TIMEZONE,
    GEMINI_API_KEY,
    GEMINI_AUDIO_MODEL,
    GEMINI_MODEL,
    GEMINI_SEARCH_MODEL,
    GEMINI_VISION_MODEL,
)


def ai_available() -> bool:
    return bool(GEMINI_API_KEY)


def _client():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    from google import genai

    return genai.Client(api_key=GEMINI_API_KEY)


def _generate_content(**kwargs: Any) -> Any:
    """Run one synchronous Gemini request with a safely owned client.

    The google-genai SDK can close the underlying HTTP client too early when
    ``genai.Client(...)`` is used only as a temporary expression, for example
    ``_client().models.generate_content(...)``. Keeping the client in a local
    context manager ensures it stays alive for the complete request and is
    then closed cleanly.
    """
    last_error: Exception | None = None

    # Retry once only for the known premature-client-close condition.
    for attempt in range(2):
        try:
            with _client() as client:
                return client.models.generate_content(**kwargs)
        except RuntimeError as exc:
            last_error = exc
            if "client has been closed" not in str(exc).lower() or attempt == 1:
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini request failed before it could be sent.")


SYSTEM_PROMPT = f"""
You are Guardian AI, a careful everyday assistant for users in {APP_COUNTRY}.
The user's timezone is {APP_TIMEZONE}. Give clear, practical answers in simple language.

Safety requirements:
- Never claim to know everything or guarantee accuracy.
- For current affairs, politics, weather, prices, law, health alerts or other changing facts, use Google Search grounding when enabled and mention the date/source context.
- Separate verified facts from uncertainty and opinion.
- Never diagnose illness or tell users to ignore a healthcare professional.
- Never claim that medicine is safe merely because it is not expired or has a registration number.
- For emergencies, advise contacting local emergency services and trusted people.
- Do not reveal, infer or help obtain private information about people from phone numbers.
- For scams, recommend independent verification through official channels and never ask for OTPs, passwords or bank PINs.
- Be politically neutral and summarize major perspectives when relevant.
- Do not invent citations or links. When Google Search grounding is used, rely on returned grounding sources.
""".strip()


def friendly_error(exc: Exception) -> str:
    message = str(exc).strip()
    lower = message.lower()

    if "api key" in lower and any(
        word in lower for word in ("invalid", "not valid", "permission")
    ):
        return (
            "Gemini rejected the API key. Create a valid key in Google AI Studio "
            "and update GEMINI_API_KEY in Streamlit Secrets."
        )
    if any(
        marker in lower
        for marker in ("429", "resource_exhausted", "quota", "rate limit")
    ):
        return (
            "The Gemini project reached a quota or rate limit. Check Google AI "
            "Studio → Dashboard → Rate limits and try again after the limit resets."
        )
    if (
        "404" in lower
        or "not found" in lower
        or ("model" in lower and "does not exist" in lower)
    ):
        return (
            "The configured Gemini model is unavailable. Set GEMINI_MODEL, "
            "GEMINI_SEARCH_MODEL, GEMINI_VISION_MODEL and GEMINI_AUDIO_MODEL "
            "to a currently available model such as gemini-3.5-flash."
        )
    if any(marker in lower for marker in ("503", "unavailable", "high demand")):
        return "Gemini is temporarily busy or unavailable. Wait briefly and try again."
    if any(marker in lower for marker in ("deadline", "timed out", "timeout")):
        return (
            "The Gemini request timed out. Try again with a shorter message or "
            "smaller media file."
        )
    if "client has been closed" in lower:
        return (
            "The Gemini connection closed unexpectedly. The app will create a fresh "
            "client for the next request; record the voice question again."
        )
    if "blocked" in lower or "safety" in lower:
        return (
            "Gemini could not return this response because of a safety restriction. "
            "Rephrase the request without unsafe details."
        )
    return message or "An unknown Gemini API error occurred."


def _history_contents(history: list[dict[str, str]]) -> list[Any]:
    from google.genai import types

    contents: list[Any] = []
    for item in history[-10:]:
        raw_role = str(item.get("role", "user")).lower()
        role = "model" if raw_role in {"assistant", "model"} else "user"
        text = str(item.get("content", "")).strip()[:8000]
        if text:
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text)],
                )
            )
    return contents


def _grounding_sources(response: Any) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        candidates = response.candidates or []
        metadata = candidates[0].grounding_metadata if candidates else None
        chunks = metadata.grounding_chunks if metadata else []
        for chunk in chunks or []:
            web = getattr(chunk, "web", None)
            uri = str(getattr(web, "uri", "") or "").strip()
            title = str(getattr(web, "title", "") or "Source").strip()
            if uri and uri not in seen:
                seen.add(uri)
                sources.append((title, uri))
    except Exception:
        return []
    return sources[:8]


def _append_sources(answer: str, response: Any) -> str:
    sources = _grounding_sources(response)
    if not sources:
        return answer
    lines = ["", "**Sources used by Gemini Google Search:**"]
    for title, uri in sources:
        safe_title = title.replace("[", "(").replace("]", ")")
        lines.append(f"- [{safe_title}]({uri})")
    return answer.rstrip() + "\n" + "\n".join(lines)


def ask_guardian(
    prompt: str,
    history: list[dict[str, str]] | None = None,
    live_search: bool = False,
    preferred_language: str | None = None,
) -> str:
    prompt = prompt.strip()
    if not prompt:
        return "Please enter a question."
    if not ai_available():
        return (
            "Gemini features are not configured yet. Add GEMINI_API_KEY to "
            "Streamlit Secrets or the local .env file. Weather, news, reminders, "
            "scam rules, QR checks and community reports can still work without it."
        )

    from google.genai import types

    system_prompt = SYSTEM_PROMPT
    if preferred_language:
        system_prompt += (
            f"\n- Reply in {preferred_language} unless the user clearly requests "
            "another language."
        )

    contents = _history_contents(history or [])
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    )

    tools = None
    model = GEMINI_MODEL
    if live_search:
        model = GEMINI_SEARCH_MODEL
        tools = [types.Tool(google_search=types.GoogleSearch())]

    try:
        response = _generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.25,
                max_output_tokens=2500,
                tools=tools,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            return "Gemini returned no text. Please try a more specific question."
        return _append_sources(answer, response) if live_search else answer
    except Exception as exc:
        return f"Gemini could not complete this request: {friendly_error(exc)}"


def _image_mime(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            fmt = (image.format or "JPEG").upper()
    except Exception:
        return "image/jpeg"
    return {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
    }.get(fmt, "image/jpeg")


def analyze_image(image_bytes: bytes, prompt: str) -> str:
    if not ai_available():
        return "Image understanding requires GEMINI_API_KEY."
    if not image_bytes:
        return "Image analysis failed: the uploaded image was empty."

    from google.genai import types

    try:
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=_image_mime(image_bytes),
        )
        response = _generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=2200,
            ),
        )
        return (response.text or "").strip() or "Gemini returned no image analysis."
    except Exception as exc:
        return f"Image analysis failed: {friendly_error(exc)}"


def analyze_image_json(image_bytes: bytes, prompt: str) -> dict[str, Any]:
    if not ai_available():
        return {"error": "Image understanding requires GEMINI_API_KEY."}
    if not image_bytes:
        return {"error": "The uploaded image was empty."}

    from google.genai import types

    raw = ""
    try:
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=_image_mime(image_bytes),
        )
        response = _generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=2200,
                response_mime_type="application/json",
            ),
        )
        raw = (response.text or "").strip()
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"raw_result": parsed}
    except json.JSONDecodeError:
        return {"raw_result": raw}
    except Exception as exc:
        return {"error": f"Image analysis failed: {friendly_error(exc)}"}


def _audio_mime(suffix: str) -> str:
    suffix = (suffix or ".wav").lower()
    manual = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    return manual.get(suffix) or mimetypes.types_map.get(suffix, "audio/wav")


def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    if not ai_available():
        return "Voice transcription requires GEMINI_API_KEY."
    if not audio_bytes:
        return "Transcription failed: the recording was empty."

    from google.genai import types

    try:
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type=_audio_mime(suffix),
        )
        prompt = (
            "Transcribe the spoken words accurately. Return only the transcript, "
            "without a summary, explanation, timestamps, speaker labels, quotation "
            "marks, or Markdown. Preserve the original spoken language."
        )
        response = _generate_content(
            model=GEMINI_AUDIO_MODEL,
            contents=[prompt, audio_part],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1200,
            ),
        )
        transcript = (response.text or "").strip()
        return transcript or "Transcription failed: Gemini returned no transcript."
    except Exception as exc:
        return f"Transcription failed: {friendly_error(exc)}"


def test_connection() -> tuple[bool, str]:
    if not ai_available():
        return False, "GEMINI_API_KEY is not configured."
    try:
        response = _generate_content(
            model=GEMINI_MODEL,
            contents="Reply with exactly: Guardian Gemini connection successful",
        )
        text = (response.text or "").strip()
        if not text:
            return False, "Gemini connected but returned no text."
        return True, f"Gemini API connected successfully using `{GEMINI_MODEL}`."
    except Exception as exc:
        return False, friendly_error(exc)
