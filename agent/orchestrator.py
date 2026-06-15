"""Orchestrator: the main agent loop.

Flow:
  1. Receive incoming message → update memories
  2. Build context from all memory layers
  3. Understanding pass → analysis + decision
  4. Decide: stay_silent | escalate | reply (auto or draft)
  5. Generate reply if needed
  6. Send or queue draft
  7. Update memories with new turn + extracted facts + pending actions
  8. Maybe consolidate (summarize old episode)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from config import settings
from schemas import IncomingMessage, MessageAnalysis, GeneratedReply, Action
from memory.manager import MemoryManager
from .understanding import Understanding
from .reply_generator import ReplyGenerator

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, memory: MemoryManager,
                 understanding: Understanding,
                 reply_gen: ReplyGenerator,
                 sender):
        """sender: callable(phone, text) -> bool (True if sent)."""
        self.memory = memory
        self.understanding = understanding
        self.reply_gen = reply_gen
        self.sender = sender

    def handle(self, msg: IncomingMessage) -> dict:
        """Process one incoming message. Returns a result dict for logging/UI."""
        phone = msg.contact_phone

        # 0. blocklist check
        if phone in settings.never_reply_contacts:
            self.memory.observe_incoming(msg)
            logger.info(f"[{phone}] in never_reply list → escalating, no reply")
            return {"action": "escalate_to_user", "reason": "contact in never_reply list"}

        # 1. ingest
        profile = self.memory.observe_incoming(msg)

        # 2. build context
        ctx = self.memory.build_context(phone, query=msg.text)

        # 3. understand
        analysis = self.understanding.analyze(
            msg=msg,
            profile=profile,
            recent_conv=ctx["recent_conversation"],
            past_episodes=ctx["past_episodes"],
            relevant_facts=ctx["relevant_facts"],
            pending_actions=ctx["pending_actions"],
        )
        logger.info(f"[{phone}] analysis: {analysis.intent} | "
                    f"{analysis.suggested_action} | urgency={analysis.urgency}")

        # 4. store extracted facts + action items + auto-update profile language/style
        if analysis.extracted_facts:
            self.memory.store_extracted_facts(phone, analysis.extracted_facts, msg.timestamp)
        if analysis.action_items:
            self.memory.store_pending_actions(phone, analysis.action_items)
        # persist detected language to profile so future replies stay consistent
        if analysis.detected_language and analysis.detected_language != "english":
            if not profile.preferred_language:
                self.memory.profile.update_profile(
                    phone, preferred_language=analysis.detected_language
                )
        # auto-set contact name if we got one and don't have it yet
        if msg.contact_name and not profile.name:
            self.memory.profile.update_profile(phone, name=msg.contact_name)

        # 5. decide what to do
        action: Action = analysis.suggested_action

        if action == "stay_silent":
            self._maybe_consolidate(phone)
            return {"action": "stay_silent", "analysis": analysis.model_dump()}

        if action == "escalate_to_user":
            # log the alert; in a real deployment you'd ping yourself via push
            logger.warning(f"[{phone}] ESCALATE: {analysis.reasoning}")
            self._maybe_consolidate(phone)
            return {
                "action": "escalate_to_user",
                "analysis": analysis.model_dump(),
                "note": "Surface this to Ankit immediately (push notification, etc.)",
            }

        # 6. generate reply
        reply = self.reply_gen.generate(
            msg=msg, profile=profile, analysis=analysis,
            recent_conv=ctx["recent_conversation"],
            past_episodes=ctx["past_episodes"],
            relevant_facts=ctx["relevant_facts"],
            pending_actions=ctx["pending_actions"],
        )
        logger.info(f"[{phone}] generated reply (conf={reply.confidence:.2f}): {reply.text[:80]}")

        # 7. send vs draft
        sent, drafted = self._send_or_draft(msg, profile, action, reply, analysis)

        # 8. update memory with our outgoing message
        self.memory.observe_outgoing(phone, reply.text, drafted=drafted, sent=sent)

        # 9. maybe consolidate
        self._maybe_consolidate(phone)

        return {
            "action": "sent" if sent else ("drafted" if drafted else "skipped"),
            "reply": reply.model_dump(),
            "analysis": analysis.model_dump(),
        }

    # ------------------------------------------------------------------

    def _send_or_draft(self, msg: IncomingMessage, profile, action: Action,
                       reply: GeneratedReply, analysis: MessageAnalysis) -> tuple[bool, bool]:
        phone = msg.contact_phone

        # forced-draft conditions
        always_draft = (
            phone in settings.always_draft_contacts
            or not settings.auto_send
            or (action == "draft_for_review" and not settings.auto_send)
            or reply.confidence < settings.auto_send_threshold
            or analysis.urgency >= 4  # high urgency → safer to review
            or not profile.auto_reply_enabled  # per-contact opt-in
        )

        if always_draft:
            self.memory.profile.enqueue_draft(
                phone=phone, name=profile.name,
                incoming=msg.text, draft=reply.text,
                rationale=reply.rationale, confidence=reply.confidence,
            )
            return (False, True)

        # auto-send path
        ok = self.sender(phone, reply.text)
        return (ok, False)

    def _maybe_consolidate(self, phone: str) -> None:
        try:
            self.memory.maybe_consolidate(phone, self.reply_gen.summarize_episode)
        except Exception as e:
            logger.exception(f"consolidation failed for {phone}: {e}")
