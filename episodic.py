"""Episodic memory: summaries of past conversation 'episodes'.

An episode = a chunk of related back-and-forth (e.g., one day's chat,
or a focused topic conversation). When short-term buffer fills up,
older turns get summarized into an episode and stored here.

Stored in SQLite + embedded in vector store for retrieval.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, DateTime, Integer, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from schemas import Episode

Base = declarative_base()


class EpisodeRow(Base):
    __tablename__ = "episodes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    contact_phone = Column(String, index=True)
    summary = Column(Text)
    key_points = Column(Text)  # json list
    start_time = Column(DateTime)
    end_time = Column(DateTime, index=True)


class EpisodicMemory:
    def __init__(self, sqlite_path: str):
        # share the same DB file as profile, separate table
        self.engine = create_engine(
            f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def add_episode(self, episode: Episode) -> int:
        with self.SessionLocal() as s:
            row = EpisodeRow(
                contact_phone=episode.contact_phone,
                summary=episode.summary,
                key_points=json.dumps(episode.key_points),
                start_time=episode.start_time,
                end_time=episode.end_time,
            )
            s.add(row)
            s.commit()
            return row.id

    def recent_episodes(self, phone: str, limit: int = 5) -> list[Episode]:
        with self.SessionLocal() as s:
            rows = (
                s.query(EpisodeRow)
                .filter(EpisodeRow.contact_phone == phone)
                .order_by(EpisodeRow.end_time.desc())
                .limit(limit)
                .all()
            )
            return [
                Episode(
                    contact_phone=r.contact_phone,
                    summary=r.summary,
                    key_points=json.loads(r.key_points or "[]"),
                    start_time=r.start_time,
                    end_time=r.end_time,
                ) for r in rows
            ]

    def render(self, phone: str, limit: int = 3) -> str:
        eps = self.recent_episodes(phone, limit)
        if not eps:
            return "(no past episodes)"
        out = []
        for e in eps:
            when = e.end_time.strftime("%Y-%m-%d")
            kp = "; ".join(e.key_points) if e.key_points else ""
            out.append(f"[{when}] {e.summary}" + (f" | key: {kp}" if kp else ""))
        return "\n".join(out)
