from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any

from config import (
    APP_COUNTRY,
    APP_TIMEZONE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TRANSCRIBE_MODEL,
    OPENAI_VISION_MODEL,
)


def ai_available() -> bool:
    return bool(OPENAI_API_KEY)


def _client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY)


SYSTEM_PROMPT = f"""
You are Guardian AI, a careful everyday assistant for users in {APP_COUNTRY}.
The user's timezone is {APP_TIMEZONE}. Give clear, practical answers in simple language.

Safety requirements:
- Never claim to know everything or guarantee accuracy.
- For current affairs, politics, weather, prices, law, health alerts or other changing facts, use live web search when available and mention the date/source context.
- Separate verified facts from uncertainty and opinion.
- Never diagnose illness or tell users to ignore a healthcare professional.
- Never claim that medicine is safe merely because it is not expired or has a registration number.
- For emergencies, advise contacting local emergency services and trusted people.
- Do not reveal, infer or help obtain private information about people from phone numbers.
- For scams, recommend independent verification through official channels and never ask for OTPs, passwords or bank PINs.
- Be politically neutral and summarize major perspectives when relevant.
""".strip()


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
            "AI features are not configured yet. Add your OPENAI_API_KEY to "
            "Streamlit Secrets or the local .env file. Weather, news, reminders, "
            "scam rules, QR checks and community reports can still work without it."
        )

    system_prompt = SYSTEM_PROMPT
    if preferred_language:
        system_prompt += (
            f"\n- Reply in {preferred_language} unless the user clearly requests "
            "another language."
        )

    history = history or []
    compact_history = history[-10:]
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for item in compact_history:
        role = item.get("role", "user")
        content = str(item.get("content", ""))[:6000]
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": messages,
    }
    if live_search:
        kwargs["tools"] = [
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": "MY",
                    "timezone": APP_TIMEZONE,
                },
            }
        ]

    try:
        response = _client().responses.create(**kwargs)
        answer = getattr(response, "output_text", "") or ""
        return answer.strip() or "The AI returned no text. Please try a more specific question."
    except Exception as exc:
        return f"The AI service could not complete this request: {exc}"


def analyze_image(image_bytes: bytes, prompt: str) -> str:
    if not ai_available():
        return "Image understanding requires OPENAI_API_KEY."
    encoded = base64.b64encode(image_bytes).decode("ascii")
    try:
        response = _client().responses.create(
            model=OPENAI_VISION_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT + "\n\n" + prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}"},
                    ],
                }
            ],
        )
        return (getattr(response, "output_text", "") or "").strip()
    except Exception as exc:
        return f"Image analysis failed: {exc}"


def analyze_image_json(image_bytes: bytes, prompt: str) -> dict[str, Any]:
    raw = analyze_image(
        image_bytes,
        prompt + "\nReturn only valid JSON. Do not use Markdown code fences.",
    )
    try:
        return json.loads(raw)
    except Exception:
        return {"raw_result": raw}


def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    if not ai_available():
        return "Voice transcription requires OPENAI_API_KEY."
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
            file.write(audio_bytes)
            path = Path(file.name)
        with path.open("rb") as audio_file:
            result = _client().audio.transcriptions.create(
                model=OPENAI_TRANSCRIBE_MODEL,
                file=audio_file,
            )
        return getattr(result, "text", str(result)).strip()
    except Exception as exc:
        return f"Transcription failed: {exc}"
    finally:
        if path:
            path.unlink(missing_ok=True)
