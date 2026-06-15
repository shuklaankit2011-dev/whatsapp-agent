"""FastAPI entry point.

Endpoints:
  POST /webhook/twilio           - Twilio WhatsApp webhook (incoming messages)
  POST /webhook/bridge           - whatsapp-web.js bridge webhook
  GET  /drafts                   - list pending drafts (your approval queue)
  POST /drafts/{id}/approve      - approve and send a draft
  POST /drafts/{id}/edit         - edit then send
  POST /drafts/{id}/reject       - reject a draft
  GET  /contacts/{phone}         - inspect a contact's profile + memory
  PATCH /contacts/{phone}        - update profile (relationship, auto-reply flag, etc.)
  GET  /health                   - health check
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Header, Request
from pydantic import BaseModel

from config import settings
from schemas import IncomingMessage
from memory.manager import MemoryManager
from agent import Understanding, ReplyGenerator, Orchestrator
from whatsapp import get_sender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("whatsapp_agent")

app = FastAPI(title="WhatsApp Reply Agent", version="0.1.0")

# wire everything up once at startup
memory = MemoryManager()
understanding = Understanding()
reply_gen = ReplyGenerator()
sender = get_sender()
orchestrator = Orchestrator(memory, understanding, reply_gen, sender.send)


# ============================================================
#  Webhooks
# ============================================================

@app.post("/webhook/twilio")
async def twilio_webhook(
    From: str = Form(...),
    Body: str = Form(...),
    ProfileName: Optional[str] = Form(None),
    MessageSid: Optional[str] = Form(None),
):
    """Twilio's WhatsApp webhook is form-urlencoded."""
    phone = From.replace("whatsapp:", "")
    msg = IncomingMessage(
        contact_phone=phone, contact_name=ProfileName, text=Body,
        message_id=MessageSid,
    )
    result = orchestrator.handle(msg)
    return {"ok": True, "result": result}


class BridgePayload(BaseModel):
    phone: str
    name: Optional[str] = None
    text: str
    message_id: Optional[str] = None
    is_group: bool = False
    group_name: Optional[str] = None
    media_type: Optional[str] = None
    media_data: Optional[str] = None


@app.post("/webhook/bridge")
async def bridge_webhook(
    payload: BridgePayload,
    authorization: Optional[str] = Header(None),
):
    """For whatsapp-web.js Node bridge. Expects Bearer token."""
    if settings.webhook_secret and authorization != f"Bearer {settings.webhook_secret}":
        raise HTTPException(401, "invalid webhook secret")
    msg = IncomingMessage(
        contact_phone=payload.phone, contact_name=payload.name, text=payload.text,
        message_id=payload.message_id, is_group=payload.is_group,
        group_name=payload.group_name, media_type=payload.media_type,
        media_data=payload.media_data,
    )
    result = orchestrator.handle(msg)
    return {"ok": True, "result": result}


# ============================================================
#  Draft approval flow
# ============================================================

@app.get("/drafts")
def list_drafts():
    return {"drafts": memory.profile.list_pending_drafts()}


class EditPayload(BaseModel):
    text: str


@app.post("/drafts/{draft_id}/approve")
def approve_draft(draft_id: int):
    drafts = memory.profile.list_pending_drafts()
    target = next((d for d in drafts if d["id"] == draft_id), None)
    if not target:
        raise HTTPException(404, "draft not found")
    ok = sender.send(target["phone"], target["draft"])
    status = "sent" if ok else "send_failed"
    memory.profile.mark_draft(draft_id, status)
    # record as sent in memory
    memory.observe_outgoing(target["phone"], target["draft"], drafted=False, sent=ok)
    return {"ok": ok, "status": status}


@app.post("/drafts/{draft_id}/edit")
def edit_draft(draft_id: int, payload: EditPayload):
    drafts = memory.profile.list_pending_drafts()
    target = next((d for d in drafts if d["id"] == draft_id), None)
    if not target:
        raise HTTPException(404, "draft not found")
    ok = sender.send(target["phone"], payload.text)
    status = "edited" if ok else "send_failed"
    memory.profile.mark_draft(draft_id, status, edited_text=payload.text)
    memory.observe_outgoing(target["phone"], payload.text, drafted=False, sent=ok)
    return {"ok": ok, "status": status, "sent_text": payload.text}


@app.post("/drafts/{draft_id}/reject")
def reject_draft(draft_id: int):
    memory.profile.mark_draft(draft_id, "rejected")
    return {"ok": True, "status": "rejected"}


# ============================================================
#  Contact management
# ============================================================

@app.get("/contacts/{phone}")
def get_contact(phone: str):
    profile = memory.profile.get_or_create_profile(phone)
    return {
        "profile": profile.model_dump(),
        "turn_count": memory.profile.turn_count(phone),
        "recent_turns": [t.model_dump() for t in memory.profile.get_turns(phone, limit=20)],
        "pending_actions": memory.profile.get_pending_actions(phone),
        "episodes": [e.model_dump() for e in memory.episodic.recent_episodes(phone, limit=5)],
        "fact_count": memory.semantic.count(phone),
    }


class ContactUpdate(BaseModel):
    name: Optional[str] = None
    relationship: Optional[str] = None
    communication_style: Optional[str] = None
    preferred_language: Optional[str] = None
    notes: Optional[str] = None
    auto_reply_enabled: Optional[bool] = None


@app.patch("/contacts/{phone}")
def update_contact(phone: str, payload: ContactUpdate):
    fields = {k: v for k, v in payload.model_dump().items() if v is not None}
    profile = memory.profile.update_profile(phone, **fields)
    return profile.model_dump()


# ============================================================
#  Health
# ============================================================

@app.get("/health")
def health():
    return {
        "ok": True,
        "provider": settings.provider,
        "auto_send": settings.auto_send,
        "model": settings.gemini_model,
        "time": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
