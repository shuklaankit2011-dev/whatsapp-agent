"""Semantic memory: atomic facts extracted from conversations, stored
in a vector store for retrieval by relevance.

Examples of facts:
- "Rahul's daughter's birthday is December 12"
- "Priya prefers calls over messages"
- "WeDezine campaign uses ad account 386459926600152"

Uses ChromaDB with Gemini embeddings.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
import google.generativeai as genai

from schemas import SemanticFact


class GeminiEmbedder:
    """Chroma-compatible embedding function using Gemini."""
    def __init__(self, model: str, api_key: str):
        genai.configure(api_key=api_key)
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        out = []
        for text in input:
            res = genai.embed_content(model=self.model, content=text)
            out.append(res["embedding"])
        return out

    def name(self) -> str:
        return f"gemini:{self.model}"


class SemanticMemory:
    def __init__(self, chroma_path: str, embed_model: str, api_key: str):
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embedder = GeminiEmbedder(embed_model, api_key)
        self.collection = self.client.get_or_create_collection(
            name="contact_facts",
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def add_fact(self, fact: SemanticFact) -> str:
        fact_id = str(uuid.uuid4())
        self.collection.add(
            ids=[fact_id],
            documents=[fact.fact],
            metadatas=[{
                "contact_phone": fact.contact_phone,
                "source_timestamp": fact.source_timestamp.isoformat(),
                "confidence": fact.confidence,
            }],
        )
        return fact_id

    def add_facts(self, facts: list[SemanticFact]) -> None:
        if not facts:
            return
        ids = [str(uuid.uuid4()) for _ in facts]
        self.collection.add(
            ids=ids,
            documents=[f.fact for f in facts],
            metadatas=[{
                "contact_phone": f.contact_phone,
                "source_timestamp": f.source_timestamp.isoformat(),
                "confidence": f.confidence,
            } for f in facts],
        )

    def recall(self, query: str, contact_phone: Optional[str] = None,
               top_k: int = 5) -> list[str]:
        """Retrieve relevant facts. If contact_phone set, filter to that contact."""
        where = {"contact_phone": contact_phone} if contact_phone else None
        try:
            res = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
            )
        except Exception:
            return []
        docs = res.get("documents", [[]])[0]
        return docs or []

    def render(self, query: str, contact_phone: Optional[str], top_k: int = 5) -> str:
        facts = self.recall(query, contact_phone, top_k)
        if not facts:
            return "(no relevant facts)"
        return "\n".join(f"- {f}" for f in facts)

    def count(self, contact_phone: Optional[str] = None) -> int:
        try:
            where = {"contact_phone": contact_phone} if contact_phone else None
            res = self.collection.get(where=where)
            return len(res.get("ids", []))
        except Exception:
            return 0
