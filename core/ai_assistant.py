from __future__ import annotations

import io
import json
import mimetypes
import re
from typing import Any

from PIL import Image

from config import (
    APP_COUNTRY,
    APP_TIMEZONE,
    GEMINI_API_KEY,
    GEMINI_AUDIO_MODEL,
    GEMINI_FALLBACK_MODELS,
    GEMINI_MODEL,
    GEMINI_SEARCH_MODEL,
    GEMINI_VISION_MODEL,
)
from core.time_utils import local_now, local_today, runtime_datetime_context


def ai_available() -> bool:
    return bool(GEMINI_API_KEY)


def _client():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    from google import genai

    return genai.Client(api_key=GEMINI_API_KEY)


def _is_fallback_error(exc: Exception) -> bool:
    lower = str(exc).lower()
    return any(
        marker in lower
        for marker in (
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "503",
            "unavailable",
            "high demand",
            "404",
            "not found",
            "does not exist",
        )
    )


def configured_models(primary_model: str | None = None) -> tuple[str, ...]:
    """Return the primary Gemini model followed by unique fallback models."""
    ordered: list[str] = []
    for model in (primary_model or GEMINI_MODEL, *GEMINI_FALLBACK_MODELS):
        clean = str(model or "").strip()
        if clean and clean not in ordered:
            ordered.append(clean)
    return tuple(ordered)


def _generate_content(*, model: str, **kwargs: Any) -> tuple[Any, str]:
    """Run Gemini with automatic model fallback."""
    models = configured_models(model)
    last_error: Exception | None = None
    attempted: list[str] = []

    for candidate in models:
        attempted.append(candidate)

        # Retry once only for the known premature-client-close condition.
        for close_attempt in range(2):
            try:
                with _client() as client:
                    response = client.models.generate_content(
                        model=candidate,
                        **kwargs,
                    )
                return response, candidate
            except Exception as exc:
                last_error = exc
                lower = str(exc).lower()

                if "client has been closed" in lower and close_attempt == 0:
                    continue

                if _is_fallback_error(exc) and candidate != models[-1]:
                    break
                raise

    attempted_text = ", ".join(attempted)
    raise RuntimeError(
        "All configured Gemini models were unavailable after automatic "
        f"fallback. Tried: {attempted_text}. Last error: {last_error}"
    )


SYSTEM_PROMPT = f"""
You are Guardian AI, a careful everyday assistant for users in {APP_COUNTRY}.
The user's timezone is {APP_TIMEZONE}. Give clear, practical answers in simple language.

Accuracy and safety requirements:
- Never claim to know everything or guarantee accuracy.
- Never use model memory alone for information that may have changed.
- Never treat an earlier assistant message as evidence.
- Distinguish verified facts, uncertainty, predictions, and opinions.
- For sports results, elections, office-holders, current affairs, weather,
  prices, laws, health alerts, schedules, releases, availability, and other
  changing facts, use live Google Search verification.
- Prefer primary official sources: tournament organizers, government agencies,
  regulators, courts, companies, universities, and original publishers.
- If reliable current evidence is unavailable, say that you could not verify
  the answer. Do not replace missing evidence with a guess.
- Never diagnose illness or tell users to ignore a healthcare professional.
- Never claim that medicine is safe merely because it is not expired or has a
  registration number.
- For emergencies, advise contacting local emergency services and trusted people.
- Do not reveal, infer or help obtain private information about people from
  phone numbers.
- For scams, recommend independent verification through official channels and
  never ask for OTPs, passwords or bank PINs.
- Be politically neutral and summarize major perspectives when relevant.
- Do not invent citations or links.
""".strip()


LIVE_VERIFICATION_INSTRUCTION = """
This request requires live verification.
- Use the Google Search tool before answering.
- Base factual claims only on the returned search evidence.
- Prefer an official primary source where one exists.
- Include the exact relevant date or year.
- Do not rely on training knowledge or an earlier answer.
- If the search evidence is incomplete or conflicting, clearly say so.
- Never say an event has not happened merely because it occurred after the
  model's training cutoff.
""".strip()


_DATE_PHRASES = (
    "what is today date",
    "what is today's date",
    "what is the date today",
    "today date",
    "today's date",
    "current date",
    "what date is it",
    "what day is it",
    "date today",
    "tarikh hari ini",
    "hari ini tarikh berapa",
    "hari ini hari apa",
    "இன்றைய தேதி",
    "இன்று என்ன தேதி",
    "இன்று என்ன நாள்",
    "今天几号",
    "今天日期",
    "今天星期几",
    "आज की तारीख",
    "आज कौन सा दिन",
    "tanggal hari ini",
    "ما تاريخ اليوم",
    "qué fecha es hoy",
    "que fecha es hoy",
    "quelle est la date aujourd'hui",
    "今日は何日",
    "오늘 날짜",
)

_TIME_PHRASES = (
    "what time is it",
    "what is the time",
    "current time",
    "time now",
    "pukul berapa sekarang",
    "masa sekarang",
    "இப்போது மணி என்ன",
    "现在几点",
    "अभी कितने बजे हैं",
    "jam berapa sekarang",
    "كم الساعة الآن",
    "qué hora es",
    "que hora es",
    "quelle heure est-il",
    "今何時",
    "지금 몇 시",
)

_CURRENT_TERMS = (
    "latest",
    "current",
    "currently",
    "today",
    "tonight",
    "yesterday",
    "tomorrow",
    "now",
    "recent",
    "recently",
    "this week",
    "this month",
    "this year",
    "breaking",
    "live",
    "update",
    "updated",
    "newest",
    "available now",
    "still available",
    "as of",
    "sekarang",
    "terkini",
    "hari ini",
    "இன்று",
    "தற்போது",
    "最新",
    "今天",
    "目前",
    "आज",
    "अभी",
    "terbaru",
    "اليوم",
    "الآن",
    "hoy",
    "actual",
    "aujourd'hui",
    "actuel",
    "今日",
    "現在",
    "오늘",
    "현재",
)

_CHANGING_FACT_TERMS = (
    "winner",
    "won",
    "who win",
    "who won",
    "champion",
    "champions",
    "score",
    "result",
    "final",
    "standings",
    "fixture",
    "schedule",
    "match",
    "world cup",
    "olympic",
    "election",
    "president",
    "prime minister",
    "minister",
    "governor",
    "mayor",
    "ceo",
    "chairman",
    "leader",
    "government",
    "politics",
    "law",
    "regulation",
    "rule",
    "policy",
    "price",
    "cost",
    "rate",
    "stock",
    "market",
    "exchange rate",
    "weather",
    "forecast",
    "warning",
    "alert",
    "outbreak",
    "recall",
    "approved",
    "banned",
    "released",
    "launch",
    "version",
    "model",
    "deadline",
    "opening hours",
    "news",
    "war",
    "conflict",
)

_SPORT_RESULT_RE = re.compile(
    r"\b("
    r"world cup|cup|league|tournament|championship|olympic|grand prix|"
    r"football|soccer|cricket|badminton|tennis|basketball|f1|formula 1"
    r")\b.*\b("
    r"won|win|winner|champion|score|result|final|beat|defeated"
    r")\b"
    r"|\b("
    r"won|win|winner|champion|score|result|final|beat|defeated"
    r")\b.*\b("
    r"world cup|cup|league|tournament|championship|olympic|grand prix|"
    r"football|soccer|cricket|badminton|tennis|basketball|f1|formula 1"
    r")\b",
    re.I,
)

_ROLE_HOLDER_RE = re.compile(
    r"\bwho\s+(is|are)\s+(the\s+)?(current\s+)?"
    r"(president|prime minister|minister|ceo|chairman|governor|mayor|leader)\b",
    re.I,
)


def _normalise_question(text: str) -> str:
    lowered = (text or "").casefold().strip()
    lowered = re.sub(r"[?？!！.,،;:]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _direct_datetime_answer(
    prompt: str,
    preferred_language: str | None = None,
) -> str | None:
    """Answer direct date/time questions locally without Gemini."""
    normalised = _normalise_question(prompt)
    asks_date = any(phrase in normalised for phrase in _DATE_PHRASES)
    asks_time = any(phrase in normalised for phrase in _TIME_PHRASES)

    if not asks_date and not asks_time:
        return None

    now = local_now()
    iso_date = now.date().isoformat()
    weekday = now.strftime("%A")
    local_time = now.strftime("%H:%M")
    language = (preferred_language or "English").casefold()

    if asks_date and asks_time:
        value = f"{iso_date} ({weekday}), {local_time}, timezone {APP_TIMEZONE}"
    elif asks_date:
        value = f"{iso_date} ({weekday}), timezone {APP_TIMEZONE}"
    else:
        value = f"{local_time} on {iso_date}, timezone {APP_TIMEZONE}"

    templates = {
        "english": "Verified local date/time: {value}.",
        "bahasa melayu": "Tarikh/masa tempatan yang disahkan: {value}.",
        "malay": "Tarikh/masa tempatan yang disahkan: {value}.",
        "tamil": "சரிபார்க்கப்பட்ட உள்ளூர் தேதி/நேரம்: {value}.",
        "chinese": "已验证的本地日期/时间：{value}。",
        "hindi": "सत्यापित स्थानीय दिनांक/समय: {value}।",
        "indonesian": "Tanggal/waktu lokal yang terverifikasi: {value}.",
        "arabic": "التاريخ/الوقت المحلي المؤكد: {value}.",
        "spanish": "Fecha/hora local verificada: {value}.",
        "french": "Date/heure locale vérifiée : {value}.",
        "japanese": "確認済みの現地日時: {value}。",
        "korean": "확인된 현지 날짜/시간: {value}.",
    }
    return templates.get(language, templates["english"]).format(value=value)


def requires_live_search(prompt: str) -> bool:
    """
    Decide whether answering from model memory would be unsafe.

    The manual search switch can request search for any question, but this
    function automatically protects current or changeable factual questions.
    """
    normalised = _normalise_question(prompt)
    if not normalised:
        return False

    if any(phrase in normalised for phrase in _DATE_PHRASES + _TIME_PHRASES):
        return False

    if _SPORT_RESULT_RE.search(normalised) or _ROLE_HOLDER_RE.search(normalised):
        return True

    has_current_term = any(term in normalised for term in _CURRENT_TERMS)
    has_changing_fact = any(term in normalised for term in _CHANGING_FACT_TERMS)
    if has_current_term and has_changing_fact:
        return True

    # Questions about the current year or a future year need verification when
    # they ask about an event, result, release, office-holder, or schedule.
    years = [int(item) for item in re.findall(r"\b(20\d{2})\b", normalised)]
    if years:
        current_year = local_today().year
        if max(years) >= current_year and has_changing_fact:
            return True
        if max(years) > current_year:
            return True

    # Explicit "has X happened", "is X still", "when will", and similar forms.
    temporal_patterns = (
        r"\bhas .+ happened\b",
        r"\bis .+ still\b",
        r"\bare .+ still\b",
        r"\bwhen (is|are|will|did)\b",
        r"\bwhat happened\b",
        r"\bwho won\b",
        r"\bwho win\b",
        r"\bdid .+ win\b",
    )
    return any(re.search(pattern, normalised) for pattern in temporal_patterns)


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
    if "all configured gemini models were unavailable" in lower:
        return (
            "Guardian AI tried every configured Gemini fallback model, but all "
            "were unavailable or at quota. Wait for the limit to reset, or add "
            "billing to the Google AI project for higher limits."
        )
    if any(
        marker in lower
        for marker in ("429", "resource_exhausted", "quota", "rate limit")
    ):
        return (
            "All configured Gemini models are currently at quota or temporarily "
            "unavailable. Guardian AI already tried its automatic fallback models. "
            "Wait for the rate limit to reset, or enable billing for higher limits."
        )
    if (
        "404" in lower
        or "not found" in lower
        or ("model" in lower and "does not exist" in lower)
    ):
        return (
            "The configured Gemini model is unavailable. Update the Gemini model "
            "names in Streamlit Secrets to models currently available to your project."
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
            "client for the next request."
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
    for item in history[-6:]:
        raw_role = str(item.get("role", "user")).lower()
        role = "model" if raw_role in {"assistant", "model"} else "user"
        text = str(item.get("content", "")).strip()[:4000]
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
        for candidate in response.candidates or []:
            metadata = getattr(candidate, "grounding_metadata", None)
            chunks = getattr(metadata, "grounding_chunks", None) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                uri = str(getattr(web, "uri", "") or "").strip()
                title = str(getattr(web, "title", "") or "Source").strip()
                if uri and uri not in seen:
                    seen.add(uri)
                    sources.append((title, uri))
    except Exception:
        return []

    return sources[:8]


def _append_verified_sources(
    answer: str,
    sources: list[tuple[str, str]],
) -> str:
    now = local_now()
    lines = [
        "",
        f"**Live verification:** {now.strftime('%Y-%m-%d %H:%M')} "
        f"({APP_TIMEZONE})",
        "",
        "**Sources used by Gemini Google Search:**",
    ]
    for title, uri in sources:
        safe_title = title.replace("[", "(").replace("]", ")")
        lines.append(f"- [{safe_title}]({uri})")
    return answer.rstrip() + "\n" + "\n".join(lines)


def _search_verification_failure() -> str:
    return (
        "I could not verify this changing information with live Google Search "
        "sources, so I will not guess or rely on an older model answer. Please "
        "try again shortly or verify it through the relevant official source."
    )


def _looks_unverified_or_stale(answer: str) -> bool:
    lower = (answer or "").casefold()
    markers = (
        "knowledge cutoff",
        "i don't have access to real-time",
        "i do not have access to real-time",
        "i cannot access current",
        "i can't access current",
        "as an ai language model",
        "the last world cup was 2022",
        "has not happened yet",
    )
    return any(marker in lower for marker in markers)


def ask_guardian(
    prompt: str,
    history: list[dict[str, str]] | None = None,
    live_search: bool = False,
    preferred_language: str | None = None,
) -> str:
    prompt = prompt.strip()
    if not prompt:
        return "Please enter a question."

    direct_datetime = _direct_datetime_answer(
        prompt,
        preferred_language=preferred_language,
    )
    if direct_datetime:
        return direct_datetime

    if not ai_available():
        return (
            "Gemini features are not configured yet. Add GEMINI_API_KEY to "
            "Streamlit Secrets. Weather, news, reminders, scam rules, QR checks "
            "and community reports can still work without it."
        )

    from google.genai import types

    search_required = bool(live_search or requires_live_search(prompt))

    system_prompt = SYSTEM_PROMPT + runtime_datetime_context()
    if preferred_language:
        system_prompt += (
            f"\n- Reply in {preferred_language} unless the user clearly requests "
            "another language."
        )
    if search_required:
        system_prompt += "\n\n" + LIVE_VERIFICATION_INSTRUCTION

    contents = _history_contents(history or [])
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        prompt
                        + (
                            "\n\nVerify this request using live Google Search and "
                            "include the exact relevant date/year."
                            if search_required
                            else ""
                        )
                    )
                )
            ],
        )
    )

    tools = None
    model = GEMINI_MODEL
    if search_required:
        model = GEMINI_SEARCH_MODEL
        tools = [types.Tool(google_search=types.GoogleSearch())]

    try:
        response, used_model = _generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1 if search_required else 0.25,
                max_output_tokens=1400,
                tools=tools,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            return "Gemini returned no text. Please try a more specific question."

        if search_required:
            sources = _grounding_sources(response)
            if not sources or _looks_unverified_or_stale(answer):
                return _search_verification_failure()
            return _append_verified_sources(answer, sources)

        return answer
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
        response, used_model = _generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1200,
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
        response, used_model = _generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=1200,
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
        response, used_model = _generate_content(
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
        response, used_model = _generate_content(
            model=GEMINI_MODEL,
            contents="Reply with exactly: Guardian Gemini connection successful",
        )
        text = (response.text or "").strip()
        if not text:
            return False, "Gemini connected but returned no text."
        return True, (
            f"Gemini API connected successfully using `{used_model}`. "
            f"Automatic fallback order: {', '.join(configured_models(GEMINI_MODEL))}."
        )
    except Exception as exc:
        return False, friendly_error(exc)
