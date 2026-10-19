"""Session management + RAG service wiring (no secrets in code)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from contextstream.config import Config
from contextstream.llm.providers import get_llm
from contextstream.rag.chunker import chunk_text
from contextstream.rag.embeddings import get_embedder
from contextstream.rag.store import build_store


@dataclass
class ChatSession:
    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    history: list[dict] = field(default_factory=list)


class RagService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = get_embedder(cfg)
        self.store = build_store(cfg)
        self.llm = get_llm(cfg)
        self.sessions: dict[str, ChatSession] = {}

    def ingest(self, doc_id: str, text: str) -> int:
        chunks = chunk_text(text, self.cfg.retrieval.chunk_max_chars,
                            self.cfg.retrieval.chunk_overlap)
        if not chunks:
            return 0
        ids = [f"{doc_id}#{i}" for i in range(len(chunks))]
        vectors = np.stack([self.embedder.embed(c) for c in chunks]).astype(np.float32)
        metadata = [{"doc_id": doc_id, "text": c} for c in chunks]
        self.store.upsert(ids, vectors, metadata)
        return len(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.cfg.retrieval.top_k
        qv = self.embedder.embed(query)
        hits = self.store.query(qv, k)
        return [{"id": h["id"], "doc_id": h["doc_id"], "score": round(h["score"], 4),
                 "text": h["text"]} for h in hits]

    def get_or_create_session(self, session_id: str | None) -> ChatSession:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        sid = session_id or uuid.uuid4().hex
        self.sessions[sid] = ChatSession(session_id=sid)
        return self.sessions[sid]

    def record(self, session: ChatSession, role: str, content: str,
               sources: list[dict] | None = None) -> None:
        entry = {"role": role, "content": content}
        if sources is not None:
            entry["sources"] = sources
        session.history.append(entry)

    async def answer(self, query: str, top_k: int | None = None):
        """Retrieve + stream. Yields (kind, payload) tuples:
        ('delta', str) ... ('final', {answer, sources, latency_ms})."""
        t0 = time.monotonic()
        hits = self.retrieve(query, top_k)
        answer_parts: list[str] = []
        async for delta in self.llm.stream(query, hits):
            answer_parts.append(delta)
            yield "delta", delta
        answer = "".join(answer_parts).strip()
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        sources = [{"id": h["id"], "doc_id": h["doc_id"], "score": h["score"]}
                   for h in hits]
        yield "final", {"answer": answer, "sources": sources, "latency_ms": latency_ms}
