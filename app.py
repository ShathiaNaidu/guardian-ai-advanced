from __future__ import annotations

import calendar
import hashlib
import html
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import streamlit as st

from config import APP_NAME, APP_TIMEZONE, GEMINI_API_KEY, GEMINI_MODEL, MALAYSIA_EMERGENCY_NUMBER, NPRA_PRODUCT_SEARCH_URL
from core.ai_assistant import (
    ai_available,
    analyze_image,
    ask_guardian,
    configured_models,
    requires_live_search,
    test_connection as test_gemini_connection,
    transcribe_audio,
)
from core.azure_voice import (
    configuration_error as azure_configuration_error,
    configured_region as azure_configured_region,
    friendly_error as azure_friendly_error,
    is_configured as azure_voice_available,
    synthesize_mp3,
    test_connection as test_azure_connection,
    transcribe_wav as azure_transcribe_wav,
    voice_for_language,
)
from core.auth import authenticate, change_password, register_user, update_profile
from core.database import execute, fetch_all, fetch_one, init_db, log_action, now_iso
from core.medicine import expiry_status, inspect_medicine_image, parse_gs1, parse_manual_medicine_text
from core.news_service import CATEGORY_QUERIES, get_news
from core.reminders import add_reminder, delete_reminder, list_reminders, set_completed
from core.safety import add_phone_report, analyze_scam_text, analyze_url, decode_qr, normalize_phone, phone_reputation
from core.weather_service import get_weather
from core.time_utils import local_now, local_today

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
css_path = BASE_DIR / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

init_db()


def init_state() -> None:
    defaults = {
        "user": None,
        "chat_history": [],
        "weather_data": None,
        "news_items": [],
        "last_ai_audio": None,
        "last_ai_audio_id": None,
        "last_autoplayed_audio_id": None,
        "voice_error": None,
        "last_processed_voice_hash": None,
        "last_voice_transcript": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def refresh_user() -> None:
    if st.session_state.user:
        fresh = fetch_one("SELECT * FROM users WHERE id = ?", (st.session_state.user["id"],))
        if fresh:
            st.session_state.user = fresh


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='guardian-hero'><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def risk_box(level: str, score: int, indicators: list[str], note: str = "") -> None:
    css_class = "risk-high" if "high" in level.lower() else "risk-medium" if "suspicious" in level.lower() or "reported" in level.lower() else "risk-low"
    points = "".join(f"<li>{html.escape(item)}</li>" for item in indicators) or "<li>No strong rule-based warning signs were detected.</li>"
    st.markdown(
        f"<div class='{css_class}'><h3>{html.escape(level)} — {score}/100</h3><ul>{points}</ul><p>{html.escape(note)}</p></div>",
        unsafe_allow_html=True,
    )


def login_page() -> None:
    hero("🛡️ Guardian AI Advanced", "A practical AI assistant for current information, safety, medicine checks, reminders and everyday support.")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Sign in")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state.user = user
                log_action(user["id"], "login", "User signed in")
                st.rerun()
            st.error("Incorrect username or password.")
    with right:
        st.subheader("Create account")
        with st.form("register_form"):
            full_name = st.text_input("Full name")
            new_username = st.text_input("Choose username")
            new_password = st.text_input("Choose password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            registered = st.form_submit_button("Create account", use_container_width=True)
        if registered:
            if new_password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message = register_user(new_username, full_name, new_password)
                (st.success if ok else st.error)(message)
    st.info("The first account created becomes the local administrator. Data is stored in this installation's SQLite database.")


def sidebar() -> str:
    user = st.session_state.user
    st.sidebar.title("🛡️ Guardian AI")
    st.sidebar.caption(f"Signed in as **{user['full_name']}**")
    if not GEMINI_API_KEY:
        st.sidebar.warning("Gemini key not configured. Core offline modules still work.")
    if not azure_voice_available():
        st.sidebar.info("Azure voice is not configured. Text replies can still work.")
    pages = [
        "Dashboard",
        "AI Assistant",
        "Live Weather",
        "Current Issues",
        "Scam Shield",
        "Medicine Check",
        "Reminders",
        "SOS & Contacts",
        "Profile",
    ]
    if user["role"] == "admin":
        pages.append("Admin")
    selected = st.sidebar.radio("Navigation", pages)
    st.sidebar.divider()
    st.sidebar.caption("Important answers should be verified through official sources. Guardian AI is an assistance tool, not an emergency, medical, legal or financial authority.")
    if st.sidebar.button("Sign out", use_container_width=True):
        log_action(user["id"], "logout", "User signed out")
        st.session_state.user = None
        st.session_state.chat_history = []
        st.session_state.last_ai_audio = None
        st.session_state.last_ai_audio_id = None
        st.session_state.last_autoplayed_audio_id = None
        st.session_state.voice_error = None
        st.session_state.last_processed_voice_hash = None
        st.session_state.last_voice_transcript = None
        st.rerun()
    return selected


def dashboard_page() -> None:
    user = st.session_state.user
    hero("Good day, " + user["full_name"], "Your safety, information and daily-assistance dashboard.")
    reminders = list_reminders(user["id"], include_completed=False)
    today = local_today()
    next_week = today + timedelta(days=7)
    due_soon = [item for item in reminders if today.isoformat() <= item["due_date"] <= next_week.isoformat()]
    reports = fetch_one("SELECT COUNT(*) AS count FROM scam_reports") or {"count": 0}
    contacts = fetch_one("SELECT COUNT(*) AS count FROM trusted_contacts WHERE user_id = ?", (user["id"],)) or {"count": 0}
    chats = fetch_one("SELECT COUNT(*) AS count FROM chat_logs WHERE user_id = ?", (user["id"],)) or {"count": 0}
    cols = st.columns(4)
    cols[0].metric("Open reminders", len(reminders))
    cols[1].metric("Due within 7 days", len(due_soon))
    cols[2].metric("Community scam reports", reports["count"])
    cols[3].metric("Trusted contacts", contacts["count"])

    st.subheader("Upcoming items")
    if due_soon:
        df = pd.DataFrame(due_soon)[["title", "category", "due_date", "due_time", "notes"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No reminders are due within the next seven days.")

    st.subheader("Guardian status")
    status_cols = st.columns(3)
    status_cols[0].success("Weather and news modules: ready")
    status_cols[1].success("Local scam and reminder database: ready")
    if ai_available() and azure_voice_available():
        status_cols[2].success("Gemini, Google Search, image/audio and Azure voice: ready")
    elif ai_available():
        status_cols[2].warning("Gemini ready; add Azure Speech secrets for spoken replies")
    else:
        status_cols[2].warning("AI features: add GEMINI_API_KEY")
    st.caption(f"You have made {chats['count']} saved AI requests in this installation.")


def _submit_ai_prompt(
    prompt: str,
    live_search: bool,
    speak: bool,
    voice_preference: str,
) -> None:
    if not prompt.strip():
        return

    user = st.session_state.user
    preferred_language = user.get("language", "English")
    history = st.session_state.chat_history
    history.append({"role": "user", "content": prompt})

    automatic_live_search = requires_live_search(prompt)
    effective_live_search = bool(live_search or automatic_live_search)

    spinner_text = (
        "Guardian AI is verifying this with live sources..."
        if effective_live_search
        else "Guardian AI is checking your request..."
    )
    with st.spinner(spinner_text):
        answer = ask_guardian(
            prompt,
            history=history[:-1],
            live_search=effective_live_search,
            preferred_language=preferred_language,
        )

    history.append({"role": "assistant", "content": answer})
    execute(
        "INSERT INTO chat_logs(user_id, prompt, response, used_live_search, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            user["id"],
            prompt[:10000],
            answer[:20000],
            1 if effective_live_search else 0,
            now_iso(),
        ),
    )

    st.session_state.voice_error = None

    if speak:
        if not azure_voice_available():
            st.session_state.last_ai_audio = None
            st.session_state.last_ai_audio_id = None
            st.session_state.voice_error = azure_configuration_error()
        else:
            try:
                audio_bytes = synthesize_mp3(
                    answer,
                    language=preferred_language,
                    voice_preference=voice_preference,
                )
                st.session_state.last_ai_audio = audio_bytes
                st.session_state.last_ai_audio_id = hashlib.sha256(audio_bytes).hexdigest()
            except Exception as exc:
                st.session_state.last_ai_audio = None
                st.session_state.last_ai_audio_id = None
                st.session_state.voice_error = azure_friendly_error(exc)
    else:
        st.session_state.last_ai_audio = None
        st.session_state.last_ai_audio_id = None

    st.rerun()


def ai_page() -> None:
    hero(
        "🤖 AI Assistant",
        "Ask Gemini with automatic model fallback, record voice questions through Azure Speech, analyze images, and hear natural Azure Neural replies.",
    )

    col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.1, 1])
    live_search = col1.toggle(
        "Use live search for every question",
        value=False,
        help=(
            "Optional. Guardian AI automatically forces live Google Search for "
            "current events, sports results, office-holders, prices, laws, "
            "weather, schedules and other changing facts even when this is off."
        ),
    )
    speak = col2.toggle(
        "Generate natural spoken reply",
        value=True,
        help="Uses Azure Neural TTS for natural spoken output.",
    )
    voice_preference = col3.selectbox(
        "Voice style",
        ["Natural female voice", "Natural male voice"],
        disabled=not speak,
    )

    if col4.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_ai_audio = None
        st.session_state.last_ai_audio_id = None
        st.session_state.last_autoplayed_audio_id = None
        st.session_state.voice_error = None
        st.session_state.last_processed_voice_hash = None
        st.session_state.last_voice_transcript = None
        st.rerun()

    with st.expander("Gemini API connection test"):
        st.caption(
            f"Primary model: `{GEMINI_MODEL}`. Automatic fallback order: "
            f"`{' → '.join(configured_models(GEMINI_MODEL))}`. Voice-note "
            "transcription uses Azure Speech first, which saves one Gemini request."
        )
        if st.button("Test Gemini API", disabled=not ai_available()):
            with st.spinner("Testing Gemini..."):
                ok, message = test_gemini_connection()
            (st.success if ok else st.error)(message)

    if speak:
        language = st.session_state.user.get("language", "English")
        if azure_voice_available():
            selected_voice = voice_for_language(language, voice_preference)
            st.success(
                f"Azure Neural TTS is ready in region "
                f"`{azure_configured_region()}` using `{selected_voice}`."
            )
        else:
            st.warning(azure_configuration_error())

        with st.expander("Azure voice connection test"):
            st.caption(
                "This checks your Azure key, region, and selected language voice. "
                "It does not expose the key."
            )
            if st.button(
                "Test Azure Neural voice",
                disabled=not azure_voice_available(),
            ):
                with st.spinner("Testing Azure Speech..."):
                    ok, message = test_azure_connection(
                        language=language,
                        voice_preference=voice_preference,
                    )
                (st.success if ok else st.error)(message)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.voice_error:
        st.error(st.session_state.voice_error)

    if st.session_state.last_ai_audio:
        audio_id = st.session_state.last_ai_audio_id
        should_autoplay = (
            bool(audio_id)
            and audio_id != st.session_state.last_autoplayed_audio_id
        )
        st.audio(
            st.session_state.last_ai_audio,
            format="audio/mpeg",
            autoplay=should_autoplay,
        )
        if should_autoplay:
            st.session_state.last_autoplayed_audio_id = audio_id
        st.caption(
            "Azure returns a normal MP3 file. Android may block the first autoplay; "
            "press Play once when needed."
        )

    st.subheader("🎤 Voice question")
    st.caption(
        "Record your question and press the recorder's stop button. Guardian AI "
        "will transcribe it with Azure Speech, send only the text question to Gemini, and create the answer."
    )
    audio = st.audio_input("Record a voice question", sample_rate=16000)

    if st.session_state.last_voice_transcript:
        st.info(f"Last voice question: {st.session_state.last_voice_transcript}")

    if audio:
        audio_bytes = audio.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest() if audio_bytes else None

        # Process each completed recording exactly once. Streamlit reruns after
        # the answer is added, while the audio widget may still contain the same
        # recording, so the hash prevents duplicate questions.
        if (
            audio_hash
            and audio_hash != st.session_state.last_processed_voice_hash
        ):
            st.session_state.last_processed_voice_hash = audio_hash
            st.session_state.voice_error = None

            preferred_language = st.session_state.user.get("language", "English")

            # Azure Speech-to-Text is attempted first so a voice question uses
            # only one Gemini request: the answer-generation request. Gemini
            # transcription remains as an emergency fallback.
            transcript = ""
            azure_transcription_error = ""
            if azure_voice_available():
                try:
                    with st.spinner("Transcribing your voice with Azure Speech..."):
                        transcript = azure_transcribe_wav(
                            audio_bytes,
                            language=preferred_language,
                        )
                except Exception as exc:
                    azure_transcription_error = azure_friendly_error(exc)

            if not transcript and ai_available():
                with st.spinner("Azure transcription was unavailable; trying Gemini..."):
                    transcript = transcribe_audio(audio_bytes, suffix=".wav")

            transcript_lower = transcript.lower()
            transcription_failed = (
                not transcript
                or transcript_lower.startswith("transcription failed")
                or "requires gemini" in transcript_lower
            )

            if transcription_failed:
                message = transcript or azure_transcription_error or (
                    "Voice transcription is unavailable. Check the Azure Speech "
                    "key and region, then try again."
                )
                st.session_state.voice_error = message
                st.error(message)
            else:
                st.session_state.last_voice_transcript = transcript
                _submit_ai_prompt(
                    transcript,
                    live_search,
                    speak,
                    voice_preference,
                )

    prompt = st.chat_input("Ask Guardian AI...")
    if prompt:
        _submit_ai_prompt(
            prompt,
            live_search,
            speak,
            voice_preference,
        )

    st.caption(
        "Automatic truth check is active: changing factual questions force live "
        "Google Search. If no grounding sources are returned, Guardian AI refuses "
        "to guess. Politics answers should remain neutral and source-based."
    )


def weather_page() -> None:
    hero("🌦️ Live Weather", "Current conditions and a seven-day forecast using Open-Meteo.")
    with st.form("weather_form"):
        city = st.text_input("City or place", value="Kuala Lumpur")
        submitted = st.form_submit_button("Get live weather")
    if submitted:
        try:
            with st.spinner("Loading forecast..."):
                st.session_state.weather_data = get_weather(city)
        except Exception as exc:
            st.error(f"Weather lookup failed: {exc}")

    data = st.session_state.weather_data
    if not data:
        return
    place = data["place"]
    current = data["current"]
    st.subheader(f"{place.get('name')}, {place.get('country', '')}")
    cols = st.columns(5)
    cols[0].metric("Temperature", f"{current.get('temperature_2m')} °C")
    cols[1].metric("Feels like", f"{current.get('apparent_temperature')} °C")
    cols[2].metric("Humidity", f"{current.get('relative_humidity_2m')}%")
    cols[3].metric("Wind", f"{current.get('wind_speed_10m')} km/h")
    cols[4].metric("Conditions", data.get("current_description", ""))

    daily = data["daily"]
    forecast = pd.DataFrame(
        {
            "Date": daily["time"],
            "Conditions": data["daily_descriptions"],
            "Minimum °C": daily["temperature_2m_min"],
            "Maximum °C": daily["temperature_2m_max"],
            "Rain chance %": daily["precipitation_probability_max"],
            "Sunrise": daily["sunrise"],
            "Sunset": daily["sunset"],
        }
    )
    st.dataframe(forecast, use_container_width=True, hide_index=True)
    fig = px.line(forecast, x="Date", y=["Minimum °C", "Maximum °C"], markers=True, title="Seven-day temperature range")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Forecasts can change. Follow official emergency warnings during floods, storms or other dangerous conditions.")


def news_page() -> None:
    hero("📰 Current Issues", "Browse recent headlines, including Malaysian politics, health, business, technology and emergencies.")
    col1, col2 = st.columns([2, 1])
    category = col1.selectbox("Topic", list(CATEGORY_QUERIES))
    limit = col2.slider("Headlines", 5, 20, 10)
    if st.button("Refresh current headlines", type="primary") or not st.session_state.news_items or st.session_state.get("news_category") != category:
        try:
            with st.spinner("Checking recent headlines..."):
                st.session_state.news_items = get_news(category, limit)
                st.session_state.news_category = category
        except Exception as exc:
            st.error(f"News feed failed: {exc}")
    items = st.session_state.news_items
    for item in items:
        title = html.escape(item["title"])
        source = html.escape(item.get("source") or "News source")
        published = html.escape(item.get("published") or "")
        link = item.get("link") or "#"
        st.markdown(f"### [{title}]({link})")
        st.caption(f"{source} · {published}")
        st.divider()
    if items and ai_available() and st.button("AI summary of these headlines"):
        headlines = "\n".join(f"- {item['title']} ({item.get('source', '')}, {item.get('published', '')})" for item in items[:12])
        summary = ask_guardian(
            "Summarize these headlines neutrally. Separate confirmed facts from claims, mention that headlines alone may omit context, and identify the main issues:\n" + headlines,
            live_search=True,
        )
        st.markdown(summary)
    st.caption("Headline feeds are starting points. Open the original report and compare multiple reputable sources, especially for political claims.")


def scam_page() -> None:
    hero("🚨 Scam Shield", "Check suspicious messages, links, QR codes and locally reported phone numbers.")
    tabs = st.tabs(["Message", "Website", "QR code", "Phone number"])
    with tabs[0]:
        text = st.text_area("Paste a suspicious SMS, WhatsApp message or email", height=180)
        col1, col2 = st.columns(2)
        if col1.button("Run scam check", use_container_width=True) and text.strip():
            result = analyze_scam_text(text)
            st.session_state.scam_text_result = result
        if col2.button("Ask AI for a second opinion", use_container_width=True) and text.strip():
            result = analyze_scam_text(text)
            prompt = (
                "Analyze the following possible scam message. Do not contact or open any link. Explain the warning signs, what should be verified, and a safe next action. "
                f"The local rule score is {result['score']}/100. Message:\n{text}"
            )
            st.session_state.scam_ai_result = ask_guardian(prompt, live_search=False)
        if "scam_text_result" in st.session_state:
            result = st.session_state.scam_text_result
            risk_box(result["level"], result["score"], result["indicators"], result["advice"])
        if "scam_ai_result" in st.session_state:
            st.markdown("#### AI explanation")
            st.markdown(st.session_state.scam_ai_result)

    with tabs[1]:
        url = st.text_input("Paste a website address", placeholder="example.com/login")
        if st.button("Analyze website address") and url.strip():
            result = analyze_url(url)
            risk_box(result["level"], result["score"], result["indicators"], result["warning"])
            st.code(result["normalized_url"])

    with tabs[2]:
        source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
        image = st.file_uploader("Upload QR image", type=["png", "jpg", "jpeg"], key="qr_upload") if source == "Upload" else st.camera_input("Photograph QR code", key="qr_camera")
        if image and st.button("Decode safely"):
            values = decode_qr(image.getvalue())
            if not values:
                st.warning("No standard QR code could be decoded. Data Matrix codes may require a dedicated scanner library or manual entry.")
            for value in values:
                st.code(value)
                if value.lower().startswith(("http://", "https://", "www.")):
                    result = analyze_url(value)
                    risk_box(result["level"], result["score"], result["indicators"], result["warning"])
                else:
                    st.info("The decoded value is displayed without opening it.")

    with tabs[3]:
        number = st.text_input("Phone number", placeholder="+60123456789")
        if st.button("Check local reputation") and number.strip():
            reputation = phone_reputation(number)
            risk_box(reputation["level"], reputation["score"], [f"{reputation['report_count']} local report(s)"] + [f"{k}: {v}" for k, v in reputation["types"].items()], reputation["note"])
            if reputation["reports"]:
                st.dataframe(pd.DataFrame(reputation["reports"]), use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("Submit a community report")
        with st.form("phone_report_form"):
            report_number = st.text_input("Reported number")
            report_type = st.selectbox("Report type", ["Impersonation", "Banking", "Investment", "Job/task", "Parcel", "Harassment", "Telemarketing", "Other"])
            description = st.text_area("What happened? Do not include passwords, OTPs or private banking details.")
            report_submit = st.form_submit_button("Submit report")
        if report_submit:
            if len(normalize_phone(report_number)) < 7:
                st.error("Enter a valid-looking phone number.")
            else:
                add_phone_report(st.session_state.user["id"], report_number, report_type, description)
                log_action(st.session_state.user["id"], "scam_report", f"Reported {normalize_phone(report_number)}")
                st.success("Report stored in this installation's community database.")
    st.caption("Guardian AI cannot identify a person's private details from a phone number and does not access telecom subscriber records.")


def medicine_page() -> None:
    hero("💊 Medicine Check", "Read visible package details, decode GS1 information and verify registration through Malaysia's official NPRA search.")
    st.warning("This tool does not diagnose, prescribe, guarantee authenticity or declare a medicine safe. Ask a pharmacist when information is missing or inconsistent.")
    tabs = st.tabs(["Scan package", "Printed text", "QR / GS1", "NPRA verification"])
    with tabs[0]:
        source = st.radio("Image source", ["Upload", "Camera"], horizontal=True, key="med_source")
        image = st.file_uploader("Upload medicine package", type=["png", "jpg", "jpeg"], key="med_upload") if source == "Upload" else st.camera_input("Photograph medicine package", key="med_camera")
        if image:
            st.image(image, caption="Medicine image", width=420)
            if st.button("Analyze visible details with AI"):
                with st.spinner("Reading only clearly visible package information..."):
                    result = inspect_medicine_image(image.getvalue())
                st.json(result)
                detected_expiry = result.get("expiry_date") if isinstance(result, dict) else None
                if detected_expiry and len(str(detected_expiry)) == 7:
                    year, month = [int(part) for part in str(detected_expiry).split("-")]
                    detected_expiry = f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
                try:
                    status = expiry_status(detected_expiry)
                    st.info(f"Expiry assessment: {status['status']} — {status['message']}")
                except Exception:
                    st.warning("The detected expiry format requires manual confirmation.")
    with tabs[1]:
        printed = st.text_area("Type or paste exactly what is printed near EXP, batch and MAL number", height=170)
        if st.button("Parse printed details") and printed.strip():
            result = parse_manual_medicine_text(printed)
            st.json(result)
    with tabs[2]:
        code = st.text_area("Paste decoded QR/Data Matrix/GS1 text", placeholder="(01)09506000123456(17)270831(10)BATCH123")
        if st.button("Decode GS1 fields") and code.strip():
            result = parse_gs1(code)
            st.json(result)
            if result.get("expiry_date"):
                status = expiry_status(result["expiry_date"])
                st.info(f"{status['status']}: {status['message']}")
            else:
                st.warning("No GS1 expiry field (17) was found. Do not estimate the expiry date.")
    with tabs[3]:
        st.markdown("Use the **official QUEST3+ Product Search** to check the MAL registration number, product name, holder, manufacturer and current status.")
        st.link_button("Open official NPRA QUEST3+ search", NPRA_PRODUCT_SEARCH_URL)
        st.markdown(
            "A matching active registration record does not independently prove that the physical package is genuine. Compare the product name, strength, manufacturer and security label, and consult a pharmacist if anything differs."
        )


def reminders_page() -> None:
    user = st.session_state.user
    hero("⏰ Reminders", "Track medicine schedules, bills, appointments and document expiry dates.")
    with st.expander("Add a reminder", expanded=True):
        with st.form("reminder_form"):
            title = st.text_input("Reminder title")
            category = st.selectbox("Category", ["Medicine", "Appointment", "Bill", "Document expiry", "Safety", "Study/work", "Other"])
            due_date = st.date_input("Due date", min_value=local_today())
            due_time = st.time_input("Time", value=local_now().replace(second=0, microsecond=0).time())
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Save reminder")
        if submitted:
            if not title.strip():
                st.error("Enter a reminder title.")
            else:
                add_reminder(user["id"], title, category, due_date.isoformat(), due_time.strftime("%H:%M"), notes)
                log_action(user["id"], "add_reminder", title)
                st.success("Reminder saved.")
                st.rerun()

    reminders = list_reminders(user["id"], include_completed=True)
    if not reminders:
        st.info("No reminders saved.")
        return
    for item in reminders:
        with st.container(border=True):
            cols = st.columns([5, 1, 1])
            status = "✅" if item["completed"] else "⏳"
            cols[0].markdown(f"### {status} {item['title']}")
            cols[0].caption(f"{item['category']} · {item['due_date']} {item['due_time'] or ''}")
            if item["notes"]:
                cols[0].write(item["notes"])
            if cols[1].button("Undo" if item["completed"] else "Done", key=f"done_{item['id']}"):
                set_completed(item["id"], user["id"], not bool(item["completed"]))
                st.rerun()
            if cols[2].button("Delete", key=f"delete_{item['id']}"):
                delete_reminder(item["id"], user["id"])
                st.rerun()
    st.caption("Streamlit reminders appear when the app is opened. True background push notifications require a mobile app or an external notification service.")


def contacts_page() -> None:
    user = st.session_state.user
    hero("🆘 SOS & Trusted Contacts", "Prepare trusted-contact actions for emergencies. The app never sends a message without the user's action.")
    st.error(f"For immediate danger in Malaysia, call {MALAYSIA_EMERGENCY_NUMBER}.")
    st.markdown(f"[📞 Call {MALAYSIA_EMERGENCY_NUMBER}](tel:{MALAYSIA_EMERGENCY_NUMBER})")
    with st.expander("Add trusted contact", expanded=True):
        with st.form("contact_form"):
            name = st.text_input("Contact name")
            number = st.text_input("Phone number", placeholder="+60123456789")
            relationship = st.text_input("Relationship")
            add_contact = st.form_submit_button("Save contact")
        if add_contact:
            normalized = normalize_phone(number)
            if not name.strip() or len(normalized) < 7:
                st.error("Enter a name and valid-looking phone number.")
            else:
                execute(
                    "INSERT INTO trusted_contacts(user_id, name, phone_number, relationship, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user["id"], name.strip(), normalized, relationship.strip(), now_iso()),
                )
                st.success("Trusted contact saved.")
                st.rerun()
    location = st.text_input("Optional location or map link to include", placeholder="Current location, landmark or Google Maps link")
    message = st.text_area("Emergency message", value="I may need help. Please contact me and check my location.")
    contacts = fetch_all("SELECT * FROM trusted_contacts WHERE user_id = ? ORDER BY name", (user["id"],))
    for contact in contacts:
        with st.container(border=True):
            cols = st.columns([4, 1])
            cols[0].markdown(f"### {contact['name']}")
            cols[0].caption(f"{contact['relationship'] or 'Trusted contact'} · {contact['phone_number']}")
            full_message = message + (f" Location: {location}" if location.strip() else "")
            digits = "".join(ch for ch in contact["phone_number"] if ch.isdigit())
            cols[0].markdown(f"[Send prepared WhatsApp message](https://wa.me/{digits}?text={quote(full_message)}) · [Call](tel:{contact['phone_number']})")
            if cols[1].button("Remove", key=f"remove_contact_{contact['id']}"):
                execute("DELETE FROM trusted_contacts WHERE id = ? AND user_id = ?", (contact["id"], user["id"]))
                st.rerun()
    st.caption("Automatic location tracking, silent SOS and background sending require a permission-based native Android/iOS implementation. Do not rely on this web prototype as the only emergency method.")


def profile_page() -> None:
    user = st.session_state.user
    hero("👤 Profile", "Manage your local Guardian AI account and preferences.")
    with st.form("profile_form"):
        full_name = st.text_input("Full name", value=user["full_name"])
        language_options = [
            "English",
            "Bahasa Melayu",
            "Tamil",
            "Chinese",
            "Hindi",
            "Indonesian",
            "Arabic",
            "Spanish",
            "French",
            "Japanese",
            "Korean",
        ]
        current_language = user.get("language", "English")
        language = st.selectbox(
            "Preferred language",
            language_options,
            index=language_options.index(current_language)
            if current_language in language_options
            else 0,
            help="Guardian AI uses this language for AI replies and Azure Neural voice selection.",
        )
        save = st.form_submit_button("Save profile")
    if save:
        update_profile(user["id"], full_name, language)
        refresh_user()
        st.success("Profile updated.")
    st.divider()
    st.subheader("Change password")
    with st.form("password_form"):
        old = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        changed = st.form_submit_button("Change password")
    if changed:
        if new != confirm:
            st.error("New passwords do not match.")
        else:
            ok, message = change_password(user["id"], old, new)
            (st.success if ok else st.error)(message)


def admin_page() -> None:
    user = st.session_state.user
    if user["role"] != "admin":
        st.error("Administrator access required.")
        return
    hero("⚙️ Administration", "Review users, reports and audit activity for this Guardian AI installation.")
    tabs = st.tabs(["Users", "Scam reports", "Audit log"])
    with tabs[0]:
        users = fetch_all("SELECT id, username, full_name, role, language, created_at FROM users ORDER BY id")
        st.dataframe(pd.DataFrame(users), use_container_width=True, hide_index=True)
        st.caption("The first registered account is automatically the administrator. Production deployments should use managed identity, MFA and stricter access controls.")
    with tabs[1]:
        reports = fetch_all("SELECT phone_number, report_type, description, created_at FROM scam_reports ORDER BY id DESC LIMIT 500")
        if reports:
            st.dataframe(pd.DataFrame(reports), use_container_width=True, hide_index=True)
            counts = pd.DataFrame(reports)["report_type"].value_counts().reset_index()
            counts.columns = ["Report type", "Count"]
            st.plotly_chart(px.bar(counts, x="Report type", y="Count", title="Community report categories"), use_container_width=True)
        else:
            st.info("No reports yet.")
    with tabs[2]:
        logs = fetch_all("SELECT user_id, action, details, created_at FROM audit_logs ORDER BY id DESC LIMIT 500")
        st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)


if not st.session_state.user:
    login_page()
    st.stop()

refresh_user()
page = sidebar()
PAGE_FUNCTIONS = {
    "Dashboard": dashboard_page,
    "AI Assistant": ai_page,
    "Live Weather": weather_page,
    "Current Issues": news_page,
    "Scam Shield": scam_page,
    "Medicine Check": medicine_page,
    "Reminders": reminders_page,
    "SOS & Contacts": contacts_page,
    "Profile": profile_page,
    "Admin": admin_page,
}
PAGE_FUNCTIONS[page]()
