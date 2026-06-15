"""MemoryManager: single entry-point that coordinates all five memory layers.

Layers:
  1. Working memory   - current turn state (in-RAM, ephemeral)
  2. Short-term       - rolling buffer of last N messages
  3. Profile          - structured contact data + full turn log (SQLite)
  4. Episodic         - summarized past episodes (SQLite)
  5. Semantic         - atomic facts vector store (ChromaDB)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from config import settings
from schemas import (
    ConversationTurn, ContactProfile, Episode, SemanticFact, IncomingMessage,
)
from .short_term import ShortTermMemory
from .profile import ProfileMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory


class MemoryManager:
    def __init__(self):
        self.short_term = ShortTermMemory(window=settings.short_term_window)
        self.profile = ProfileMemory(sqlite_path=settings.sqlite_path)
        self.episodic = EpisodicMemory(sqlite_path=settings.sqlite_path)
        self.semantic = SemanticMemory(
            chroma_path=settings.chroma_path,
            embed_model=settings.gemini_embed_model,
            api_key=settings.gemini_api_key,
        )

    # ---------- ingestion ----------

    def observe_incoming(self, msg: IncomingMessage) -> ContactProfile:
        """Called when a message arrives. Updates short-term + profile + turn log."""
        profile = self.profile.get_or_create_profile(msg.contact_phone, msg.contact_name)
        turn = ConversationTurn(
            contact_phone=msg.contact_phone, role="contact",
            text=msg.text, timestamp=msg.timestamp,
        )
        self.short_term.append(turn)
        self.profile.log_turn(turn)
        self.profile.update_profile(msg.contact_phone, last_seen=msg.timestamp)
        return profile

    def observe_outgoing(self, phone: str, text: str, drafted: bool, sent: bool) -> None:
        turn = ConversationTurn(
            contact_phone=phone, role="agent" if drafted else "user",
            text=text, was_drafted=drafted, was_sent=sent,
        )
        self.short_term.append(turn)
        self.profile.log_turn(turn)

    def store_extracted_facts(self, phone: str, facts: list[str], when: datetime) -> None:
        if not facts:
            return
        self.semantic.add_facts([
            SemanticFact(contact_phone=phone, fact=f, source_timestamp=when)
            for f in facts
        ])

    def store_pending_actions(self, phone: str, actions: list[str]) -> None:
        for a in actions:
            self.profile.add_pending_action(phone, a)

    # ---------- retrieval (assembles context for the agent) ----------

    def build_context(self, phone: str, query: str) -> dict:
        """Assemble all memory layers into a context dict for prompts."""
        profile = self.profile.get_or_create_profile(phone)
        return {
            "profile": profile,
            "recent_conversation": self.short_term.render(phone),
            "past_episodes": self.episodic.render(phone, limit=6),
            "relevant_facts": self.semantic.render(query, phone, top_k=10),
            "pending_actions": self.profile.get_pending_actions(phone),
            "turn_count": self.profile.turn_count(phone),
        }

    # ---------- consolidation ----------

    def maybe_consolidate(self, phone: str, summarizer) -> Optional[int]:
        """If short-term is full, summarize older half into an episode.

        `summarizer` is a callable: (turns: list[ConversationTurn]) -> (summary, key_points)
        Returns episode_id if one was created.
        """
        if self.short_term.size(phone) < settings.short_term_window:
            return None

        turns = self.short_term.recent(phone)
        # take the older half to summarize
        half = len(turns) // 2
        to_summarize = turns[:half]
        if not to_summarize:
            return None

        summary, key_points = summarizer(to_summarize)
        episode = Episode(
            contact_phone=phone,
            summary=summary,
            key_points=key_points,
            start_time=to_summarize[0].timestamp,
            end_time=to_summarize[-1].timestamp,
        )
        ep_id = self.episodic.add_episode(episode)

        # also store as a semantic fact for retrievability
        self.semantic.add_facts([
            SemanticFact(
                contact_phone=phone,
                fact=f"Past conversation summary: {summary}",
                source_timestamp=episode.end_time,
            )
        ])

        # trim short-term: keep only the newer half
        # (we rebuild the deque)
        keep = turns[half:]
        self.short_term.clear(phone)
        for t in keep:
            self.short_term.append(t)
        return ep_id
