"""Chunking documents into overlapping windows."""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 500, overlap: int = 80) -> list[str]:
    """Chunk into <=max_chars windows, preferring sentence boundaries, with
    character overlap between consecutive chunks."""
    if max_chars <= overlap:
        raise ValueError("max_chars must exceed overlap")
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT.split(text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(sent) > max_chars:
            for i in range(0, len(sent), max_chars - overlap):
                piece = sent[i : i + max_chars]
                if current:
                    chunks.append(current)
                    current = current[-overlap:] if overlap else ""
                current = (current + " " + piece).strip() if current else piece
                if len(current) >= max_chars:
                    chunks.append(current)
                    current = current[-overlap:] if overlap else ""
            continue
        if len(current) + len(sent) + 1 > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + " " + sent).strip()
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]
