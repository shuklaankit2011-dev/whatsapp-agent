"""WhatsApp connector. Two providers supported:

1. "twilio" — Twilio's WhatsApp Business API (official, paid, requires verified number)
2. "bridge" — your own whatsapp-web.js Node.js bridge (free, unofficial, against ToS technically)

Both expose the same `send(phone, text) -> bool` interface.

Incoming messages are received via the FastAPI webhook in main.py and normalized
into IncomingMessage objects, so the orchestrator doesn't care which provider sent them.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)


class WhatsAppClient:
    """Abstract sender. Subclasses implement .send()."""
    def send(self, phone: str, text: str) -> bool:
        raise NotImplementedError


class TwilioClient(WhatsAppClient):
    def __init__(self):
        from twilio.rest import Client
        self.client = Client(settings.twilio_sid, settings.twilio_token)
        self.from_ = settings.twilio_from  # e.g. "whatsapp:+14155238886"

    def send(self, phone: str, text: str) -> bool:
        try:
            to = phone if phone.startswith("whatsapp:") else f"whatsapp:{phone}"
            self.client.messages.create(from_=self.from_, to=to, body=text)
            return True
        except Exception as e:
            logger.exception(f"Twilio send failed for {phone}: {e}")
            return False


class BridgeClient(WhatsAppClient):
    """Talks to a whatsapp-web.js Node bridge (see bridge.js example below)."""
    def __init__(self):
        self.url = settings.bridge_url
        self.token = settings.bridge_token

    def send(self, phone: str, text: str) -> bool:
        try:
            r = httpx.post(
                self.url,
                json={"phone": phone, "text": text},
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=15,
            )
            r.raise_for_status()
            return True
        except Exception as e:
            logger.exception(f"Bridge send failed for {phone}: {e}")
            return False


def get_sender() -> WhatsAppClient:
    if settings.provider == "twilio":
        return TwilioClient()
    if settings.provider == "bridge":
        return BridgeClient()
    raise ValueError(f"Unknown WHATSAPP_PROVIDER: {settings.provider}")
