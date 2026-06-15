"""Short-term memory: rolling buffer of last N messages per contact.

This is the agent's 'working context' — what was *just* said.
Kept in-memory; persisted snapshot in SQLite handled by ProfileMemory.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque

from schemas import ConversationTurn


class ShortTermMemory:
    def __init__(self, window: int = 15):
        self.window = window
        self._buffers: dict[str, Deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def append(self, turn: ConversationTurn) -> None:
        self._buffers[turn.contact_phone].append(turn)

    def recent(self, contact_phone: str, n: int | None = None) -> list[ConversationTurn]:
        buf = list(self._buffers.get(contact_phone, []))
        if n is None:
            return buf
        return buf[-n:]

    def render(self, contact_phone: str, n: int | None = None) -> str:
        """Render as text for LLM prompt."""
        turns = self.recent(contact_phone, n)
        if not turns:
            return "(no recent messages)"
        lines = []
        for t in turns:
            who = {"contact": "Them", "user": "Me", "agent": "Agent"}[t.role]
            lines.append(f"{who}: {t.text}")
        return "\n".join(lines)

    def clear(self, contact_phone: str) -> None:
        self._buffers.pop(contact_phone, None)

    def size(self, contact_phone: str) -> int:
        return len(self._buffers.get(contact_phone, []))
