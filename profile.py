"""Profile memory: long-term structured facts about each contact.

Backed by SQLite. Stores:
- Contact profile (name, relationship, comm style, auto-reply flag)
- Full turn log (every message ever exchanged)
- Pending action items
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, Text, create_engine, select
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from schemas import ContactProfile, ConversationTurn

Base = declarative_base()


class ContactRow(Base):
    __tablename__ = "contacts"
    phone = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    relationship = Column(String, nullable=True)
    communication_style = Column(String, nullable=True)
    preferred_language = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    auto_reply_enabled = Column(Boolean, default=False)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)


class TurnRow(Base):
    __tablename__ = "turns"
    id = Column(Integer, primary_key=True, autoincrement=True)
    contact_phone = Column(String, index=True)
    role = Column(String)
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    was_drafted = Column(Boolean, default=False)
    was_sent = Column(Boolean, default=False)


class PendingAction(Base):
    __tablename__ = "pending_actions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    contact_phone = Column(String, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)


class DraftQueue(Base):
    __tablename__ = "draft_queue"
    id = Column(Integer, primary_key=True, autoincrement=True)
    contact_phone = Column(String, index=True)
    contact_name = Column(String, nullable=True)
    incoming_text = Column(Text)
    draft_text = Column(Text)
    rationale = Column(Text)
    confidence = Column(String)  # store as text for simplicity
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")  # pending | sent | rejected | edited


class ProfileMemory:
    def __init__(self, sqlite_path: str):
        self.engine = create_engine(
            f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    # ---------- profile ----------

    def get_or_create_profile(self, phone: str, name: Optional[str] = None) -> ContactProfile:
        with self.SessionLocal() as s:
            row = s.get(ContactRow, phone)
            if row is None:
                row = ContactRow(phone=phone, name=name)
                s.add(row)
                s.commit()
            elif name and not row.name:
                row.name = name
                s.commit()
            return self._row_to_profile(row)

    def update_profile(self, phone: str, **fields) -> ContactProfile:
        with self.SessionLocal() as s:
            row = s.get(ContactRow, phone)
            if row is None:
                row = ContactRow(phone=phone)
                s.add(row)
            for k, v in fields.items():
                if hasattr(row, k) and v is not None:
                    setattr(row, k, v)
            row.last_seen = datetime.utcnow()
            s.commit()
            return self._row_to_profile(row)

    @staticmethod
    def _row_to_profile(row: ContactRow) -> ContactProfile:
        return ContactProfile(
            phone=row.phone,
            name=row.name,
            relationship=row.relationship,
            communication_style=row.communication_style,
            preferred_language=row.preferred_language,
            notes=row.notes,
            auto_reply_enabled=row.auto_reply_enabled or False,
            first_seen=row.first_seen or datetime.utcnow(),
            last_seen=row.last_seen or datetime.utcnow(),
        )

    # ---------- turns ----------

    def log_turn(self, turn: ConversationTurn) -> None:
        with self.SessionLocal() as s:
            s.add(TurnRow(
                contact_phone=turn.contact_phone,
                role=turn.role,
                text=turn.text,
                timestamp=turn.timestamp,
                was_drafted=turn.was_drafted,
                was_sent=turn.was_sent,
            ))
            s.commit()

    def turn_count(self, phone: str) -> int:
        with self.SessionLocal() as s:
            return s.query(TurnRow).filter(TurnRow.contact_phone == phone).count()

    def get_turns(self, phone: str, limit: int = 50) -> list[ConversationTurn]:
        with self.SessionLocal() as s:
            rows = (
                s.query(TurnRow)
                .filter(TurnRow.contact_phone == phone)
                .order_by(TurnRow.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                ConversationTurn(
                    contact_phone=r.contact_phone, role=r.role, text=r.text,
                    timestamp=r.timestamp, was_drafted=r.was_drafted or False,
                    was_sent=r.was_sent or False,
                ) for r in reversed(rows)
            ]

    # ---------- pending actions ----------

    def add_pending_action(self, phone: str, description: str) -> None:
        with self.SessionLocal() as s:
            s.add(PendingAction(contact_phone=phone, description=description))
            s.commit()

    def get_pending_actions(self, phone: str) -> list[str]:
        with self.SessionLocal() as s:
            rows = (
                s.query(PendingAction)
                .filter(PendingAction.contact_phone == phone, PendingAction.resolved == False)  # noqa
                .all()
            )
            return [r.description for r in rows]

    # ---------- draft queue ----------

    def enqueue_draft(self, phone: str, name: Optional[str], incoming: str,
                      draft: str, rationale: str, confidence: float) -> int:
        with self.SessionLocal() as s:
            d = DraftQueue(
                contact_phone=phone, contact_name=name,
                incoming_text=incoming, draft_text=draft,
                rationale=rationale, confidence=str(confidence),
            )
            s.add(d)
            s.commit()
            return d.id

    def list_pending_drafts(self) -> list[dict]:
        with self.SessionLocal() as s:
            rows = (
                s.query(DraftQueue)
                .filter(DraftQueue.status == "pending")
                .order_by(DraftQueue.created_at.desc())
                .all()
            )
            return [
                {
                    "id": r.id, "phone": r.contact_phone, "name": r.contact_name,
                    "incoming": r.incoming_text, "draft": r.draft_text,
                    "rationale": r.rationale, "confidence": r.confidence,
                    "created_at": r.created_at.isoformat(),
                } for r in rows
            ]

    def mark_draft(self, draft_id: int, status: str, edited_text: Optional[str] = None) -> Optional[dict]:
        with self.SessionLocal() as s:
            r = s.get(DraftQueue, draft_id)
            if r is None:
                return None
            r.status = status
            if edited_text is not None:
                r.draft_text = edited_text
            s.commit()
            return {"id": r.id, "phone": r.contact_phone, "draft": r.draft_text, "status": r.status}
