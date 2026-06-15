"""Pydantic schemas for messages, analysis, replies, and memory records."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------- WhatsApp I/O ----------

class IncomingMessage(BaseModel):
    """Normalized incoming message from any WhatsApp provider."""
    contact_phone: str
    contact_name: Optional[str] = None
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_group: bool = False
    group_name: Optional[str] = None
    message_id: Optional[str] = None
    media_url: Optional[str] = None
    media_type: Optional[str] = None   # e.g. "image/jpeg"
    media_data: Optional[str] = None   # base64-encoded bytes


class OutgoingReply(BaseModel):
    contact_phone: str
    text: str
    in_reply_to: Optional[str] = None


# ---------- Understanding ----------

Intent = Literal[
    "question", "request", "social_chat", "scheduling",
    "information_share", "complaint", "urgent", "spam", "other"
]
Emotion = Literal["neutral", "happy", "sad", "angry", "anxious", "excited", "frustrated"]
Action = Literal["reply_now", "draft_for_review", "stay_silent", "escalate_to_user"]


class MessageAnalysis(BaseModel):
    intent: Intent
    emotion: Emotion
    urgency: int = Field(ge=1, le=5, description="1=trivial, 5=critical")
    topic: str = Field(description="Short topic phrase, e.g. 'project deadline'")
    requires_my_knowledge: bool = Field(
        description="True if reply needs facts only Ankit knows (calendar, opinions, decisions)"
    )
    action_items: list[str] = Field(default_factory=list)
    extracted_facts: list[str] = Field(
        default_factory=list,
        description="Atomic facts about the contact worth remembering"
    )
    detected_language: str = Field(
        default="english",
        description="Language of the incoming message: english, hindi, hinglish, or other"
    )
    image_description: Optional[str] = Field(
        default=None,
        description="What the image shows, if a media message was sent"
    )
    suggested_action: Action
    reasoning: str


class GeneratedReply(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    rationale: str


# ---------- Memory records ----------

class ConversationTurn(BaseModel):
    contact_phone: str
    role: Literal["contact", "user", "agent"]  # contact=them, user=Ankit, agent=auto-reply
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    was_drafted: bool = False
    was_sent: bool = False


class ContactProfile(BaseModel):
    phone: str
    name: Optional[str] = None
    relationship: Optional[str] = None  # "client", "friend", "family", "colleague", "vendor", "unknown"
    communication_style: Optional[str] = None  # "formal", "casual", "hinglish", etc.
    preferred_language: Optional[str] = None
    notes: Optional[str] = None
    auto_reply_enabled: bool = True
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)


class Episode(BaseModel):
    """A summarized chunk of past conversation."""
    contact_phone: str
    summary: str
    start_time: datetime
    end_time: datetime
    key_points: list[str] = Field(default_factory=list)


class SemanticFact(BaseModel):
    """An atomic fact extracted from conversation."""
    contact_phone: str
    fact: str
    source_timestamp: datetime
    confidence: float = 1.0
