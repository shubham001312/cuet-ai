from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_NAME = "cuet_AI"
APP_VERSION = "2.0-web"
TODAY = dt.date.today()
ACTIVE_CUET_YEAR = TODAY.year
TODAY_TEXT = TODAY.strftime("%B %d, %Y")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SERVER_CONFIG_PATH = Path.home() / ".cuet_ai" / "config.json"


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    return max(value, minimum)


PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip() or "gemini-2.5-flash-lite"
MODEL_FALLBACKS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
MAX_OUTPUT_TOKENS = env_int("CUET_AI_MAX_OUTPUT_TOKENS", 1000, minimum=128)
MAX_CONTEXT_MESSAGES = env_int("CUET_AI_CONTEXT_MESSAGES", 8, minimum=1)
MAX_STORED_MESSAGES = env_int("CUET_AI_STORED_MESSAGES", 30, minimum=2)
RATE_LIMIT_WINDOW_SECONDS = env_int("CUET_AI_RATE_LIMIT_WINDOW_SECONDS", 3600, minimum=60)
RATE_LIMIT_MAX_REQUESTS = env_int("CUET_AI_RATE_LIMIT_MAX_REQUESTS", 20, minimum=1)
MODEL_CANDIDATES: list[str] = []
for candidate in [PRIMARY_MODEL, *MODEL_FALLBACKS]:
    if candidate and candidate not in MODEL_CANDIDATES:
        MODEL_CANDIDATES.append(candidate)

OFFICIAL_SOURCES = [
    "https://cuet.nta.nic.in/",
    "https://www.nta.ac.in/",
    "https://exams.nta.ac.in/CUET-UG/",
]

QUICK_QUERIES = [
    {
        "label": "Latest Official Notices",
        "query": (
            f"What are the latest official notices for CUET UG {ACTIVE_CUET_YEAR}? "
            "Give exact dates, what changed, and the official links."
        ),
    },
    {
        "label": "Exam Dates",
        "query": (
            f"What are the official CUET UG {ACTIVE_CUET_YEAR} exam dates and shifts? "
            "List exact dates if announced and say clearly if anything is still unannounced."
        ),
    },
    {
        "label": "Admit Card",
        "query": (
            f"Has the CUET UG {ACTIVE_CUET_YEAR} admit card been released? "
            "Explain the exact download steps and official portal."
        ),
    },
    {
        "label": "City Intimation Slip",
        "query": (
            f"How do I check the CUET UG {ACTIVE_CUET_YEAR} city intimation slip? "
            "Give the official site and exact steps."
        ),
    },
    {
        "label": "Registration",
        "query": (
            f"Summarize the CUET UG {ACTIVE_CUET_YEAR} registration process, fees, "
            "important deadlines, and correction-window details."
        ),
    },
    {
        "label": "Syllabus & Pattern",
        "query": (
            f"What is the CUET UG {ACTIVE_CUET_YEAR} syllabus and exam pattern? "
            "Explain sections, number of questions, marking scheme, and any major changes."
        ),
    },
]

TIMELINE = [
    {
        "title": "Registration Window",
        "note": f"Verify live form status for CUET UG {ACTIVE_CUET_YEAR} on the official portal.",
        "state": "watch",
    },
    {
        "title": "Correction Facility",
        "note": "NTA typically opens limited edit windows after form submission. Check notices carefully.",
        "state": "watch",
    },
    {
        "title": "City Intimation Slip",
        "note": "This appears before the admit card. Treat unofficial screenshots as unverified.",
        "state": "current",
    },
    {
        "title": "Admit Card Download",
        "note": "Use only the official candidate login. Verify exam date, shift, and reporting time.",
        "state": "current",
    },
    {
        "title": "Exam Window",
        "note": f"Track subject-wise scheduling for CUET UG {ACTIVE_CUET_YEAR}; sessions can differ by candidate.",
        "state": "current",
    },
    {
        "title": "Result & Counselling",
        "note": "After results, follow each university's admission portal separately.",
        "state": "watch",
    },
]

ALERTS = [
    "Always verify critical announcements on cuet.nta.nic.in or nta.ac.in.",
    "Never trust admit-card or correction links shared only through social media forwards.",
    "Use exact dates from official notices, not expected timelines from coaching channels.",
]

WELCOME_MESSAGE = f"""cuet_AI is ready.

Ask about:
- CUET UG {ACTIVE_CUET_YEAR} exam dates and shifts
- Admit card release and download steps
- Registration deadlines, fee rules, and correction windows
- Syllabus, pattern, result timing, and counselling

Paste your Google AI API key above, or set GEMINI_API_KEY in the terminal before launching.
Always cross-check high-stakes information on cuet.nta.nic.in and nta.ac.in.
"""

SYSTEM_PROMPT = f"""You are cuet_AI, an expert assistant for CUET UG (Common University Entrance Test - Undergraduate) in India.

Your job:
- Focus on the latest official CUET UG information.
- Prioritize official sources: cuet.nta.nic.in, nta.ac.in, and exams.nta.ac.in/CUET-UG.
- Give exact dates when they are officially available.
- If something is not officially announced, say so plainly.
- For admit-card, city-slip, result, and counselling questions, give clear step-by-step instructions.
- Keep answers practical for students.

Rules:
1. Do not invent dates, deadlines, or portal links.
2. If information may have changed, say students should verify on the official websites.
3. Highlight urgent actions clearly.
4. Prefer concise sections with bullet points.
5. When possible, include official URLs in the answer.

Today's date is {TODAY_TEXT}.
"""

SESSION_COOKIE = "cuet_ai_session"
SESSION_LOCK = threading.Lock()
SESSIONS: dict[str, dict[str, Any]] = {}
RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_BUCKETS: dict[str, list[float]] = {}


class GeminiAPIError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def load_server_config() -> dict[str, Any]:
    try:
        if not SERVER_CONFIG_PATH.exists():
            return {}
        return json.loads(SERVER_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_server_api_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    config = load_server_config()
    return str(config.get("api_key", "")).strip()


def apply_rate_limit(client_id: str) -> tuple[bool, int]:
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_BUCKETS.get(client_id, [])
        bucket = [timestamp for timestamp in bucket if timestamp >= cutoff]
        allowed = len(bucket) < RATE_LIMIT_MAX_REQUESTS
        if allowed:
            bucket.append(now)
        RATE_LIMIT_BUCKETS[client_id] = bucket
        remaining = max(RATE_LIMIT_MAX_REQUESTS - len(bucket), 0)
        return allowed, remaining


def build_request_history(session_history: list[dict[str, str]], new_message: str) -> list[dict[str, str]]:
    history_slice = session_history[-MAX_CONTEXT_MESSAGES:] if MAX_CONTEXT_MESSAGES > 0 else []
    return list(history_slice) + [{"role": "user", "content": new_message}]


def new_session() -> dict[str, Any]:
    return {
        "history": [],
        "updates": 0,
        "tokens": 0,
        "last_search": "",
        "last_model": "",
    }


def get_or_create_session(session_id: str) -> dict[str, Any]:
    with SESSION_LOCK:
        session = SESSIONS.get(session_id)
        if session is None:
            session = new_session()
            SESSIONS[session_id] = session
        return session


def reset_session(session_id: str) -> dict[str, Any]:
    with SESSION_LOCK:
        SESSIONS[session_id] = new_session()
        return SESSIONS[session_id]


def public_state(session: dict[str, Any], server_has_key: bool) -> dict[str, Any]:
    return {
        "history": list(session["history"]),
        "updates": session["updates"],
        "tokens": session["tokens"],
        "lastSearch": session["last_search"],
        "lastModel": session["last_model"],
        "hasServerApiKey": server_has_key,
        "welcome": WELCOME_MESSAGE,
    }


def should_try_next_model(status_code: int, message: str) -> bool:
    lowered = message.lower()
    model_markers = [
        "model",
        "not found",
        "unsupported",
        "not available",
        "not found for api version",
        "unknown",
    ]
    return status_code == 404 or (status_code == 400 and any(marker in lowered for marker in model_markers))


def build_gemini_contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = "user" if message["role"] == "user" else "model"
        contents.append(
            {
                "role": role,
                "parts": [{"text": message["content"]}],
            }
        )
    return contents


def generate_with_model(api_key: str, model: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int], str]:
    payload = json.dumps(
        {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT}],
            },
            "contents": build_gemini_contents(messages),
            "generationConfig": {
                "temperature": 0.55,
                "topK": 32,
                "topP": 0.92,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL.format(model=model) + f"?key={api_key}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            details = json.loads(body)
            message = details.get("error", {}).get("message", body)
        except json.JSONDecodeError:
            message = body or str(error)
        raise GeminiAPIError(message=message, status_code=error.code) from error
    except urllib.error.URLError as error:
        raise GeminiAPIError(message=f"Network error: {error.reason}", status_code=503) from error

    candidates = data.get("candidates", [])
    if not candidates:
        blocked = data.get("promptFeedback", {}).get("blockReason")
        if blocked:
            raise GeminiAPIError(message=f"Request blocked by API safety policy: {blocked}", status_code=400)
        raise GeminiAPIError(message="The API returned no response text.", status_code=502)

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise GeminiAPIError(message="The API returned an empty response.", status_code=502)

    usage_meta = data.get("usageMetadata", {})
    input_tokens = int(usage_meta.get("promptTokenCount") or len(payload) // 4)
    output_tokens = int(usage_meta.get("candidatesTokenCount") or max(len(text) // 4, 1))
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(usage_meta.get("totalTokenCount") or (input_tokens + output_tokens)),
    }
    return text, usage, model


def generate_response(api_key: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int], str]:
    errors: list[str] = []
    for model in MODEL_CANDIDATES:
        try:
            return generate_with_model(api_key=api_key, model=model, messages=messages)
        except GeminiAPIError as error:
            errors.append(f"{model}: {error.message}")
            if should_try_next_model(error.status_code, error.message):
                continue
            raise
    raise GeminiAPIError(message=" ; ".join(errors), status_code=400)


def render_home(bootstrap: dict[str, Any]) -> bytes:
    bootstrap_json = json.dumps(bootstrap, ensure_ascii=False).replace("</", "<\\/")
    html = INDEX_HTML.replace("__BOOTSTRAP_JSON__", bootstrap_json)
    return html.encode("utf-8")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>cuet_AI</title>
  <style>
    :root {
      --bg: #08131b;
      --panel: rgba(8, 24, 34, 0.82);
      --panel-strong: rgba(10, 29, 41, 0.96);
      --border: rgba(110, 188, 220, 0.18);
      --text: #ecf7fb;
      --muted: #88a9b7;
      --dim: #5f7a88;
      --aqua: #72f2d3;
      --cyan: #57b7ff;
      --amber: #f4bb64;
      --red: #ff8578;
      --shadow: 0 20px 60px rgba(0, 0, 0, 0.34);
      --radius: 22px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 15% 15%, rgba(87, 183, 255, 0.16), transparent 24rem),
        radial-gradient(circle at 85% 10%, rgba(114, 242, 211, 0.12), transparent 26rem),
        linear-gradient(160deg, #071018 0%, #091824 42%, #08131b 100%);
      color: var(--text);
      font-family: Bahnschrift, "Aptos", "Segoe UI", sans-serif;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
      background-size: 34px 34px;
      mask-image: linear-gradient(to bottom, rgba(0,0,0,0.7), transparent);
      opacity: 0.4;
    }

    .shell {
      width: min(1420px, calc(100vw - 32px));
      margin: 24px auto 32px;
      position: relative;
      z-index: 1;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 18px;
      align-items: stretch;
      padding: 18px;
      border: 1px solid var(--border);
      border-radius: 28px;
      background:
        linear-gradient(140deg, rgba(8, 24, 34, 0.94), rgba(8, 24, 34, 0.62)),
        linear-gradient(90deg, rgba(114, 242, 211, 0.06), transparent);
      backdrop-filter: blur(18px);
      box-shadow: var(--shadow);
    }

    .brand {
      display: flex;
      gap: 16px;
      align-items: center;
      min-width: 0;
    }

    .brand-mark {
      flex: 0 0 auto;
      width: 68px;
      height: 68px;
      border-radius: 20px;
      display: grid;
      place-items: center;
      font: 700 1.4rem/1 "Cascadia Code", Consolas, monospace;
      color: #041016;
      background: linear-gradient(140deg, var(--aqua), #b8fff0);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.4), 0 18px 34px rgba(114, 242, 211, 0.25);
    }

    .eyebrow {
      margin: 0 0 6px;
      color: var(--aqua);
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font: 700 0.72rem/1 "Cascadia Code", Consolas, monospace;
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.2rem);
      line-height: 0.95;
      letter-spacing: -0.05em;
    }

    .lede {
      margin: 10px 0 0;
      max-width: 60ch;
      color: var(--muted);
      font-size: 1rem;
    }

    .hero-right {
      display: grid;
      gap: 12px;
      min-width: 0;
    }

    .key-panel,
    .micro-panel {
      border: 1px solid var(--border);
      border-radius: 20px;
      background: rgba(5, 17, 24, 0.72);
      padding: 14px 16px;
    }

    .key-label,
    .panel-label {
      display: block;
      margin: 0 0 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font: 700 0.68rem/1 "Cascadia Code", Consolas, monospace;
    }

    .key-row,
    .compose-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }

    input,
    textarea,
    button {
      font: inherit;
    }

    input,
    textarea {
      width: 100%;
      border: 1px solid rgba(114, 242, 211, 0.12);
      background: rgba(4, 14, 20, 0.88);
      color: var(--text);
      border-radius: 16px;
      outline: none;
      transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    }

    input {
      height: 48px;
      padding: 0 14px;
    }

    textarea {
      min-height: 86px;
      resize: vertical;
      padding: 14px;
      line-height: 1.5;
    }

    input:focus,
    textarea:focus {
      border-color: rgba(114, 242, 211, 0.4);
      box-shadow: 0 0 0 3px rgba(114, 242, 211, 0.08);
    }

    button {
      border: 0;
      border-radius: 16px;
      padding: 0 16px;
      min-height: 48px;
      color: #08131b;
      background: linear-gradient(135deg, var(--aqua), #c9fff2);
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.18s ease, filter 0.18s ease, opacity 0.18s ease;
    }

    button:hover { transform: translateY(-1px); filter: brightness(1.04); }
    button:disabled { cursor: wait; opacity: 0.62; transform: none; }

    .ghost {
      color: var(--text);
      background: rgba(87, 183, 255, 0.14);
      border: 1px solid rgba(87, 183, 255, 0.22);
    }

    .danger {
      color: #ffd7d1;
      background: rgba(255, 133, 120, 0.14);
      border: 1px solid rgba(255, 133, 120, 0.22);
    }

    .micro {
      margin: 8px 0 0;
      color: var(--dim);
      font-size: 0.9rem;
      min-height: 1.25rem;
    }

    .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 38px;
      padding: 0 14px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(4, 14, 20, 0.84);
      color: var(--muted);
      font: 700 0.78rem/1 "Cascadia Code", Consolas, monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .pill::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--cyan);
      box-shadow: 0 0 16px rgba(87, 183, 255, 0.7);
    }

    .pill.live::before { background: var(--aqua); box-shadow: 0 0 16px rgba(114, 242, 211, 0.75); }
    .pill.alert::before { background: var(--amber); box-shadow: 0 0 16px rgba(244, 187, 100, 0.75); }

    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0;
    }

    .stat {
      padding: 16px;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .stat-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font: 700 0.72rem/1 "Cascadia Code", Consolas, monospace;
    }

    .stat-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--aqua);
      box-shadow: 0 0 18px rgba(114, 242, 211, 0.75);
    }

    .stat-value {
      margin: 12px 0 6px;
      font-size: clamp(1.5rem, 2.4vw, 2.2rem);
      line-height: 1;
      letter-spacing: -0.05em;
    }

    .stat-sub {
      color: var(--dim);
      font-size: 0.92rem;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.65fr) minmax(300px, 0.92fr);
      gap: 16px;
      align-items: start;
    }

    .panel {
      border: 1px solid var(--border);
      border-radius: 26px;
      background: var(--panel-strong);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 18px 18px 0;
    }

    .panel-head h2,
    .sidebar h3 {
      margin: 0;
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--aqua);
      font-family: "Cascadia Code", Consolas, monospace;
    }

    .panel-subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.95rem;
    }

    .alert-banner {
      margin: 16px 18px 0;
      padding: 12px 14px;
      border-radius: 18px;
      border: 1px solid rgba(244, 187, 100, 0.18);
      background: linear-gradient(135deg, rgba(244, 187, 100, 0.15), rgba(244, 187, 100, 0.04));
      color: #ffdca4;
    }

    .chat-log {
      padding: 18px;
      min-height: 430px;
      max-height: 68vh;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .empty-state {
      padding: 20px;
      border-radius: 20px;
      background: linear-gradient(135deg, rgba(87, 183, 255, 0.1), rgba(114, 242, 211, 0.06));
      border: 1px solid rgba(87, 183, 255, 0.14);
      color: var(--muted);
      white-space: pre-wrap;
      line-height: 1.6;
    }

    .message {
      align-self: stretch;
      display: flex;
      gap: 12px;
      align-items: flex-start;
    }

    .message.user { justify-content: flex-end; }

    .avatar {
      flex: 0 0 auto;
      width: 36px;
      height: 36px;
      border-radius: 12px;
      display: grid;
      place-items: center;
      font: 700 0.82rem/1 "Cascadia Code", Consolas, monospace;
      border: 1px solid var(--border);
      background: rgba(87, 183, 255, 0.12);
      color: var(--cyan);
    }

    .message.user .avatar {
      background: rgba(114, 242, 211, 0.12);
      color: var(--aqua);
    }

    .bubble {
      max-width: min(760px, 100%);
      padding: 14px 16px;
      border-radius: 20px;
      background: rgba(7, 17, 23, 0.94);
      border: 1px solid rgba(255,255,255,0.06);
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .message.user .bubble {
      background: linear-gradient(135deg, rgba(21, 76, 92, 0.92), rgba(10, 44, 53, 0.94));
    }

    .message.error .bubble {
      border-color: rgba(255, 133, 120, 0.24);
      background: rgba(41, 14, 14, 0.92);
      color: #ffd1cb;
    }

    .bubble-meta {
      margin-top: 10px;
      color: var(--dim);
      font: 700 0.76rem/1 "Cascadia Code", Consolas, monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .bubble strong { color: #fff4df; }
    .bubble a { color: #8fd6ff; }

    .composer {
      padding: 0 18px 18px;
    }

    .compose-row {
      align-items: end;
    }

    .send-btn {
      min-width: 132px;
    }

    .sidebar {
      display: grid;
      gap: 14px;
    }

    .side-card {
      padding: 16px;
    }

    .quick-grid,
    .source-list,
    .alert-list,
    .timeline-list {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }

    .quick-btn {
      width: 100%;
      justify-content: space-between;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 14px 16px;
      border-radius: 18px;
      color: var(--text);
      background: rgba(7, 17, 23, 0.82);
      border: 1px solid rgba(255,255,255,0.06);
      text-align: left;
    }

    .quick-btn span:last-child {
      color: var(--dim);
      font: 700 0.76rem/1 "Cascadia Code", Consolas, monospace;
    }

    .timeline-item,
    .source-item,
    .alert-item {
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(7, 17, 23, 0.74);
      padding: 12px 14px;
    }

    .timeline-head {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }

    .timeline-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--muted);
      box-shadow: 0 0 0 transparent;
    }

    .timeline-dot.current {
      background: var(--amber);
      box-shadow: 0 0 14px rgba(244, 187, 100, 0.72);
    }

    .timeline-dot.watch {
      background: var(--cyan);
      box-shadow: 0 0 14px rgba(87, 183, 255, 0.72);
    }

    .timeline-title {
      font-weight: 700;
    }

    .timeline-note,
    .source-url,
    .alert-item {
      color: var(--muted);
      line-height: 1.5;
    }

    .source-label {
      color: var(--text);
      font-weight: 700;
      margin-bottom: 6px;
    }

    .footer {
      margin-top: 14px;
      text-align: center;
      color: var(--dim);
      font-size: 0.9rem;
    }

    @media (max-width: 1080px) {
      .hero,
      .layout {
        grid-template-columns: 1fr;
      }

      .stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 720px) {
      .shell { width: min(100vw - 16px, 100%); margin: 12px auto 24px; }
      .hero,
      .stat,
      .panel,
      .side-card { border-radius: 20px; }
      .stats { grid-template-columns: 1fr; }
      .key-row,
      .compose-row { grid-template-columns: 1fr; }
      .send-btn,
      .ghost,
      .danger { width: 100%; }
      .chat-log { min-height: 360px; max-height: none; }
      .panel-head { align-items: flex-start; flex-direction: column; }
      .message { gap: 10px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="hero">
      <div class="brand">
        <div class="brand-mark">CA</div>
        <div>
          <p class="eyebrow">CUET local browser build</p>
          <h1>cuet_AI</h1>
          <p class="lede">A web version of your desktop monitor with official-link-first answers, quick CUET actions, and local-session history.</p>
        </div>
      </div>
      <div class="hero-right">
        <section class="key-panel">
          <label class="key-label" for="apiKeyInput">Google AI API Key</label>
          <div class="key-row">
            <input id="apiKeyInput" type="password" placeholder="AIza..." autocomplete="off">
            <button id="saveKeyBtn" class="ghost" type="button">Save Key</button>
          </div>
          <p id="keyStatus" class="micro"></p>
        </section>
        <section class="micro-panel">
          <div class="status-row">
            <span class="pill live">Model <span id="modelPill">...</span></span>
            <span class="pill alert">Session <span id="sessionPill">browser key required</span></span>
          </div>
        </section>
      </div>
    </header>

    <section class="stats">
      <article class="stat">
        <div class="stat-head"><span>Official Sources</span><span class="stat-dot"></span></div>
        <div class="stat-value" id="sourcesValue">0</div>
        <div class="stat-sub">NTA and CUET portals in the watchlist</div>
      </article>
      <article class="stat">
        <div class="stat-head"><span>Queries This Session</span><span class="stat-dot"></span></div>
        <div class="stat-value" id="queriesValue">0</div>
        <div class="stat-sub">Local browser session history</div>
      </article>
      <article class="stat">
        <div class="stat-head"><span>Token Usage</span><span class="stat-dot"></span></div>
        <div class="stat-value" id="tokensValue">0</div>
        <div class="stat-sub">Estimated or API-reported token count</div>
      </article>
      <article class="stat">
        <div class="stat-head"><span>Tracking Year</span><span class="stat-dot"></span></div>
        <div class="stat-value" id="yearValue">----</div>
        <div class="stat-sub">Ask for exact official dates, not expected ones</div>
      </article>
    </section>

    <section class="layout">
      <main class="panel">
        <div class="panel-head">
          <div>
            <h2>AI Intelligence Assistant</h2>
            <p class="panel-subtitle">Browser chat for admit cards, city slips, dates, syllabus, results, and counselling.</p>
          </div>
          <button id="resetBtn" type="button" class="danger">Reset Chat</button>
        </div>
        <div class="alert-banner">Use exact official dates and portals. If something is not announced yet, the answer should say that clearly.</div>
        <div id="chatLog" class="chat-log"></div>
        <form id="chatForm" class="composer">
          <label class="panel-label" for="messageInput">Ask cuet_AI</label>
          <div class="compose-row">
            <textarea id="messageInput" placeholder="Ask about admit card release, exam date, city slip, syllabus, result timeline, or counselling..."></textarea>
            <button id="sendBtn" class="send-btn" type="submit">Ask AI</button>
          </div>
          <p id="requestStatus" class="micro"></p>
        </form>
      </main>

      <aside class="sidebar">
        <section class="panel side-card">
          <h3>Quick Queries</h3>
          <div id="quickGrid" class="quick-grid"></div>
        </section>

        <section class="panel side-card">
          <h3>Timeline Watch</h3>
          <div id="timelineList" class="timeline-list"></div>
        </section>

        <section class="panel side-card">
          <h3>Priority Alerts</h3>
          <div id="alertList" class="alert-list"></div>
        </section>

        <section class="panel side-card">
          <h3>Official Sources</h3>
          <div id="sourceList" class="source-list"></div>
        </section>
      </aside>
    </section>

    <p class="footer">Running locally on your machine. Browser-saved keys stay in local storage unless you set <code>GEMINI_API_KEY</code> before launching.</p>
  </div>

  <script>
    const BOOTSTRAP = __BOOTSTRAP_JSON__;

    const apiKeyInput = document.getElementById("apiKeyInput");
    const saveKeyBtn = document.getElementById("saveKeyBtn");
    const keyStatus = document.getElementById("keyStatus");
    const modelPill = document.getElementById("modelPill");
    const sessionPill = document.getElementById("sessionPill");
    const sourcesValue = document.getElementById("sourcesValue");
    const queriesValue = document.getElementById("queriesValue");
    const tokensValue = document.getElementById("tokensValue");
    const yearValue = document.getElementById("yearValue");
    const chatLog = document.getElementById("chatLog");
    const chatForm = document.getElementById("chatForm");
    const messageInput = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const requestStatus = document.getElementById("requestStatus");
    const resetBtn = document.getElementById("resetBtn");
    const quickGrid = document.getElementById("quickGrid");
    const timelineList = document.getElementById("timelineList");
    const alertList = document.getElementById("alertList");
    const sourceList = document.getElementById("sourceList");

    let state = BOOTSTRAP.state;

    function escapeHtml(text) {
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatMessage(text) {
      let safe = escapeHtml(text);
      safe = safe.replace(/\\*\\*\\*(.+?)\\*\\*\\*/gs, "<strong>$1</strong>");
      safe = safe.replace(/\\*\\*(.+?)\\*\\*/gs, "<strong>$1</strong>");
      safe = safe.replace(/(https?:\\/\\/[^\\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
      return safe.replace(/\\n/g, "<br>");
    }

    function getStoredKey() {
      return localStorage.getItem("cuet_ai_api_key") || "";
    }

    function setRequestStatus(message, isError = false) {
      requestStatus.textContent = message || "";
      requestStatus.style.color = isError ? "#ffb4ac" : "";
    }

    function refreshKeyStatus() {
      const stored = getStoredKey();
      apiKeyInput.value = stored;

      if (state.hasServerApiKey) {
        keyStatus.textContent = "Server API key is configured. The browser field is optional.";
        sessionPill.textContent = "server key ready";
      } else if (stored) {
        keyStatus.textContent = "Browser API key saved locally for this machine.";
        sessionPill.textContent = "browser key ready";
      } else {
        keyStatus.textContent = "Paste a Google AI API key here or set GEMINI_API_KEY before launch.";
        sessionPill.textContent = "browser key required";
      }
    }

    function renderStats() {
      sourcesValue.textContent = String(BOOTSTRAP.officialSources.length);
      queriesValue.textContent = String(state.updates || 0);
      tokensValue.textContent = Number(state.tokens || 0).toLocaleString();
      yearValue.textContent = "CUET UG " + String(BOOTSTRAP.year);
      modelPill.textContent = state.lastModel || BOOTSTRAP.models[0] || "default";
    }

    function renderQuickQueries() {
      quickGrid.innerHTML = "";
      BOOTSTRAP.quickQueries.forEach((item, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "quick-btn";
        button.innerHTML = "<span>" + escapeHtml(item.label) + "</span><span>Q" + String(index + 1) + "</span>";
        button.addEventListener("click", () => {
          messageInput.value = item.query;
          messageInput.focus();
          sendMessage(item.query);
        });
        quickGrid.appendChild(button);
      });
    }

    function renderTimeline() {
      timelineList.innerHTML = "";
      BOOTSTRAP.timeline.forEach((item) => {
        const wrapper = document.createElement("div");
        wrapper.className = "timeline-item";
        wrapper.innerHTML =
          '<div class="timeline-head">' +
            '<span class="timeline-dot ' + escapeHtml(item.state || "watch") + '"></span>' +
            '<div class="timeline-title">' + escapeHtml(item.title) + '</div>' +
          '</div>' +
          '<div class="timeline-note">' + escapeHtml(item.note) + '</div>';
        timelineList.appendChild(wrapper);
      });
    }

    function renderAlerts() {
      alertList.innerHTML = "";
      BOOTSTRAP.alerts.forEach((item) => {
        const wrapper = document.createElement("div");
        wrapper.className = "alert-item";
        wrapper.textContent = item;
        alertList.appendChild(wrapper);
      });
    }

    function renderSources() {
      sourceList.innerHTML = "";
      BOOTSTRAP.officialSources.forEach((item) => {
        const wrapper = document.createElement("div");
        wrapper.className = "source-item";
        wrapper.innerHTML =
          '<div class="source-label">' + escapeHtml(item.label) + '</div>' +
          '<div class="source-url"><a href="' + escapeHtml(item.url) + '" target="_blank" rel="noopener noreferrer">' +
          escapeHtml(item.url) +
          "</a></div>";
        sourceList.appendChild(wrapper);
      });
    }

    function appendMessage(role, text, meta = "") {
      const wrapper = document.createElement("div");
      wrapper.className = "message " + role;

      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = role === "user" ? "YOU" : role === "error" ? "ERR" : "AI";

      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.innerHTML = formatMessage(text);

      if (meta) {
        const metaNode = document.createElement("div");
        metaNode.className = "bubble-meta";
        metaNode.textContent = meta;
        bubble.appendChild(metaNode);
      }

      if (role === "user") {
        wrapper.appendChild(bubble);
        wrapper.appendChild(avatar);
      } else {
        wrapper.appendChild(avatar);
        wrapper.appendChild(bubble);
      }

      chatLog.appendChild(wrapper);
    }

    function renderChat() {
      chatLog.innerHTML = "";

      if (!state.history || state.history.length === 0) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.innerHTML = formatMessage(state.welcome);
        chatLog.appendChild(empty);
      } else {
        state.history.forEach((message, index) => {
          const isLatestAssistant =
            message.role === "assistant" &&
            index === state.history.length - 1 &&
            Boolean(state.lastModel || state.lastSearch);

          let meta = "";
          if (isLatestAssistant) {
            const parts = [];
            if (state.lastModel) {
              parts.push("Model: " + state.lastModel);
            }
            if (state.lastSearch) {
              parts.push("Updated: " + state.lastSearch);
            }
            meta = parts.join("  •  ");
          }

          appendMessage(message.role, message.content, meta);
        });
      }

      chatLog.scrollTop = chatLog.scrollHeight;
    }

    function renderState() {
      renderStats();
      renderChat();
      refreshKeyStatus();
    }

    function toggleBusy(isBusy) {
      sendBtn.disabled = isBusy;
      resetBtn.disabled = isBusy;
      messageInput.disabled = isBusy;
      sendBtn.textContent = isBusy ? "Searching..." : "Ask AI";
    }

    function saveBrowserKey() {
      const key = apiKeyInput.value.trim();
      if (key && !key.startsWith("AIza")) {
        keyStatus.textContent = "Google AI keys usually start with AIza. Check the pasted key.";
        keyStatus.style.color = "#ffb4ac";
        return;
      }

      keyStatus.style.color = "";

      if (key) {
        localStorage.setItem("cuet_ai_api_key", key);
      } else {
        localStorage.removeItem("cuet_ai_api_key");
      }

      refreshKeyStatus();
    }

    async function resetChat() {
      toggleBusy(true);
      setRequestStatus("Resetting session...");

      try {
        const response = await fetch("/api/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({})
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Could not reset the session.");
        }
        state = data.state;
        renderState();
        setRequestStatus("Session reset.");
      } catch (error) {
        appendMessage("error", error.message || "Could not reset the session.");
        setRequestStatus(error.message || "Could not reset the session.", true);
      } finally {
        toggleBusy(false);
      }
    }

    async function sendMessage(forcedMessage = "") {
      const message = (forcedMessage || messageInput.value || "").trim();
      if (!message) {
        return;
      }

      const apiKey = getStoredKey();
      if (!state.hasServerApiKey && !apiKey) {
        setRequestStatus("An API key is required before you can send a query.", true);
        apiKeyInput.focus();
        return;
      }

      toggleBusy(true);
      setRequestStatus("Searching and generating a response...");

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: message,
            apiKey: apiKey
          })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "The request failed.");
        }

        state = data.state;
        messageInput.value = "";
        renderState();
        setRequestStatus(state.lastSearch ? "Last update: " + state.lastSearch : "Response ready.");
      } catch (error) {
        appendMessage("error", error.message || "The request failed.");
        setRequestStatus(error.message || "The request failed.", true);
      } finally {
        toggleBusy(false);
      }
    }

    saveKeyBtn.addEventListener("click", saveBrowserKey);
    resetBtn.addEventListener("click", resetChat);
    chatForm.addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage();
    });

    messageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });

    renderQuickQueries();
    renderTimeline();
    renderAlerts();
    renderSources();
    renderState();
    setRequestStatus(state.lastSearch ? "Last update: " + state.lastSearch : "Ready.");
  </script>
</body>
</html>
"""


def build_bootstrap(session: dict[str, Any], server_has_key: bool) -> dict[str, Any]:
    return {
        "appName": APP_NAME,
        "version": APP_VERSION,
        "year": ACTIVE_CUET_YEAR,
        "models": MODEL_CANDIDATES,
        "quickQueries": QUICK_QUERIES,
        "timeline": TIMELINE,
        "alerts": ALERTS,
        "officialSources": [
            {"label": "CUET Official Portal", "url": OFFICIAL_SOURCES[0]},
            {"label": "NTA Main Website", "url": OFFICIAL_SOURCES[1]},
            {"label": "NTA Exam Page", "url": OFFICIAL_SOURCES[2]},
        ],
        "state": public_state(session, server_has_key),
    }


class CuetAIHandler(BaseHTTPRequestHandler):
    server_version = f"{APP_NAME}/{APP_VERSION}"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        session_id = self._ensure_session()
        session = get_or_create_session(session_id)
        server_has_key = self._server_has_key()

        if path == "/":
            payload = render_home(build_bootstrap(session, server_has_key))
            self._send_bytes(payload, content_type="text/html; charset=utf-8")
            return

        if path == "/api/state":
            self._send_json(build_bootstrap(session, server_has_key))
            return

        if path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "app": APP_NAME,
                    "version": APP_VERSION,
                    "models": MODEL_CANDIDATES,
                }
            )
            return

        self._send_json({"error": "Not found."}, status=404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        session_id = self._ensure_session()
        server_has_key = self._server_has_key()

        if path == "/api/chat":
            try:
                body = self._read_json_body()
                message = str(body.get("message", "")).strip()
                browser_key = str(body.get("apiKey", "")).strip()
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
                return

            if not message:
                self._send_json({"error": "Message cannot be empty."}, status=400)
                return

            if len(message) > 6000:
                self._send_json({"error": "Message is too long. Keep it under 6000 characters."}, status=400)
                return

            shared_server_key = get_server_api_key()
            api_key = browser_key or shared_server_key
            if not api_key:
                self._send_json({"error": "No API key provided. Paste one in the browser or set GEMINI_API_KEY."}, status=400)
                return

            if not api_key.startswith("AIza"):
                self._send_json({"error": "Google AI API keys usually start with AIza. Check the key and try again."}, status=400)
                return

            if shared_server_key and api_key == shared_server_key:
                allowed, remaining = apply_rate_limit(self._client_identifier())
                if not allowed:
                    wait_minutes = max(RATE_LIMIT_WINDOW_SECONDS // 60, 1)
                    self._send_json(
                        {
                            "error": (
                                f"Shared server quota reached for this client. Try again later, "
                                f"or paste your own API key. Window: {wait_minutes} minute(s)."
                            )
                        },
                        status=429,
                    )
                    return

            session = get_or_create_session(session_id)
            request_history = build_request_history(session["history"], message)

            try:
                text, usage, model = generate_response(api_key=api_key, messages=request_history)
            except GeminiAPIError as error:
                self._send_json({"error": error.message}, status=error.status_code)
                return

            with SESSION_LOCK:
                session["history"].append({"role": "user", "content": message})
                session["history"].append({"role": "assistant", "content": text})
                if len(session["history"]) > MAX_STORED_MESSAGES:
                    session["history"] = session["history"][-MAX_STORED_MESSAGES:]
                session["updates"] += 1
                session["tokens"] += int(usage.get("total_tokens", 0))
                session["last_search"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                session["last_model"] = model
                response_state = public_state(session, server_has_key)

            self._send_json({"ok": True, "state": response_state})
            return

        if path == "/api/reset":
            session = reset_session(session_id)
            self._send_json({"ok": True, "state": public_state(session, server_has_key)})
            return

        self._send_json({"error": "Not found."}, status=404)

    def _ensure_session(self) -> str:
        self._cookie_to_set = ""
        cookie_header = self.headers.get("Cookie", "")
        cookies = SimpleCookie()
        if cookie_header:
            try:
                cookies.load(cookie_header)
            except Exception:
                cookies = SimpleCookie()

        existing = cookies.get(SESSION_COOKIE)
        if existing and existing.value:
            return existing.value

        session_id = uuid.uuid4().hex
        self._cookie_to_set = f"{SESSION_COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Lax"
        return session_id

    def _server_has_key(self) -> bool:
        return bool(get_server_api_key())

    def _client_identifier(self) -> str:
        forwarded = self.headers.get("X-Forwarded-For", "").strip()
        if forwarded:
            return forwarded.split(",", 1)[0].strip() or self.client_address[0]
        return self.client_address[0]

    def _read_json_body(self) -> dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length header.") from error

        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON.") from error

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, status=status, content_type="application/json; charset=utf-8")

    def _send_bytes(self, payload: bytes, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if getattr(self, "_cookie_to_set", ""):
            self.send_header("Set-Cookie", self._cookie_to_set)
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cuet_AI as a local browser app.")
    parser.add_argument("--host", default=os.getenv("HOST", os.getenv("CUET_AI_HOST", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", os.getenv("CUET_AI_PORT", "8000"))))
    parser.add_argument("--open-browser", action="store_true", help="Open the local app in the default browser.")
    return parser.parse_args()


def maybe_open_browser(url: str, enabled: bool) -> None:
    if not enabled:
        return

    timer = threading.Timer(0.9, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def main() -> None:
    args = parse_args()
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{display_host}:{args.port}/"

    server = ThreadingHTTPServer((args.host, args.port), CuetAIHandler)
    print(f"{APP_NAME} running at {url}")
    print(f"Model order: {', '.join(MODEL_CANDIDATES)}")
    print(
        f"Public guardrails: {RATE_LIMIT_MAX_REQUESTS} shared-key requests per {RATE_LIMIT_WINDOW_SECONDS}s "
        f"per client, {MAX_CONTEXT_MESSAGES} context messages, {MAX_OUTPUT_TOKENS} max output tokens."
    )
    if get_server_api_key():
        print("Server API key detected from environment or local config.")
    else:
        print("No server API key detected. Paste a Google AI API key into the browser UI.")

    maybe_open_browser(url, args.open_browser)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping cuet_AI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
