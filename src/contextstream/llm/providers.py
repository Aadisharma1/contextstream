"""LLM providers: local mock (extractive, streaming) + external via env."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from contextstream.rag.embeddings import _tokenize


class MockLLM:
    """Extractive 'generation': picks the query-relevant sentences from the
    retrieved context and streams them. Real streaming mechanics, honest
    about being a mock: it composes answers from the retrieved chunks only."""

    def __init__(self, stream_delay_ms: int = 5):
        self.stream_delay_ms = stream_delay_ms

    async def stream(self, query: str, context_chunks: list[dict]) -> AsyncIterator[str]:
        q_tokens = set(_tokenize(query))
        scored: list[tuple[float, str]] = []
        for chunk in context_chunks:
            for sentence in _sentence_iter(chunk["text"]):
                s_tokens = set(_tokenize(sentence))
                overlap = len(q_tokens & s_tokens) / max(len(s_tokens), 1)
                scored.append((overlap, sentence))
        scored.sort(key=lambda t: -t[0])
        answer_sentences = [s for overlap, s in scored if overlap > 0][:3]
        if not answer_sentences:
            answer = ("I could not find anything relevant in the indexed documents "
                      "for that question.")
        else:
            answer = " ".join(answer_sentences)

        words = answer.split()
        for i in range(0, len(words), 3):
            yield " ".join(words[i : i + 3]) + " "
            await asyncio.sleep(self.stream_delay_ms / 1000.0)


class ExternalLLM:
    """OpenAI-compatible streaming chat provider. Config via env only.
    CONTEXTSTREAM_API_BASE + CONTEXTSTREAM_API_KEY (+ CONTEXTSTREAM_LLM_MODEL)."""

    def __init__(self, api_base: str, api_key: str, model: str):
        if not api_base or not api_key:
            raise RuntimeError(
                "ExternalLLM requires CONTEXTSTREAM_API_BASE and "
                "CONTEXTSTREAM_API_KEY env vars (no hardcoded keys)."
            )
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def stream(self, query: str, context_chunks: list[dict]) -> AsyncIterator[str]:
        import httpx

        context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in context_chunks)
        payload = {
            "model": self.model,
            "stream": True,
            "messages": [
                {"role": "system", "content": "Answer using the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json

                        delta = json.loads(line[6:])["choices"][0]["delta"]
                        if delta.get("content"):
                            yield delta["content"]


def _sentence_iter(text: str):
    import re

    for s in re.split(r"(?<=[.!?])\s+", text):
        if s.strip():
            yield s.strip()


def get_llm(cfg):
    if cfg.llm.provider == "external":
        import os

        return ExternalLLM(
            os.environ.get("CONTEXTSTREAM_API_BASE", ""),
            os.environ.get("CONTEXTSTREAM_API_KEY", ""),
            os.environ.get("CONTEXTSTREAM_LLM_MODEL", "gpt-4o-mini"),
        )
    return MockLLM(cfg.llm.stream_delay_ms)
