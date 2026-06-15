"""Configuration loaded from environment."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _csv(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    # LLM
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

    # User
    user_name: str = os.getenv("USER_NAME", "User")
    user_phone: str = os.getenv("USER_PHONE", "")
    persona_file: str = os.getenv("USER_PERSONA_FILE", "persona.md")

    # Behavior
    auto_send: bool = os.getenv("AUTO_SEND", "false").lower() == "true"
    auto_send_threshold: float = float(os.getenv("AUTO_SEND_CONFIDENCE_THRESHOLD", "0.85"))
    always_draft_contacts: tuple[str, ...] = tuple(_csv(os.getenv("ALWAYS_DRAFT_CONTACTS", "")))
    never_reply_contacts: tuple[str, ...] = tuple(_csv(os.getenv("NEVER_REPLY_CONTACTS", "")))

    # Storage
    sqlite_path: str = os.getenv("SQLITE_PATH", "./data/memory.db")
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma")
    short_term_window: int = int(os.getenv("SHORT_TERM_WINDOW", "15"))

    # WhatsApp
    provider: str = os.getenv("WHATSAPP_PROVIDER", "bridge")
    twilio_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_from: str = os.getenv("TWILIO_WHATSAPP_FROM", "")
    bridge_url: str = os.getenv("BRIDGE_SEND_URL", "")
    bridge_token: str = os.getenv("BRIDGE_AUTH_TOKEN", "")

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "")


settings = Settings()

# Ensure data dirs exist
Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
