"""Embedding providers.

`local`: deterministic hashing bag-of-tokens projection — no network, no API
key, stable across runs (real, just simple). `external`: OpenAI-compatible
/embeddings endpoint configured ONLY through env vars; never hardcoded keys.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


def _tokenize(text: str) -> list[str]:
    return "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class LocalHashEmbedder:
    """Deterministic hashed bag-of-tokens embedding, L2-normalized."""

    def __init__(self, dim: int = 256):
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            h = hashlib.md5(tok.encode()).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class ExternalEmbedder:
    """OpenAI-compatible /embeddings client. Config via env only."""

    def __init__(self, api_base: str, api_key: str, dim: int, model: str = "text-embedding-3-small"):
        if not api_base or not api_key:
            raise RuntimeError(
                "ExternalEmbedder requires CONTEXTSTREAM_API_BASE and "
                "CONTEXTSTREAM_API_KEY environment variables (no hardcoded keys)."
            )
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.dim = dim
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        import httpx

        r = httpx.post(
            f"{self.api_base}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": text[:8000]},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()["data"][0]["embedding"]
        arr = np.asarray(data, dtype=np.float32)
        norm = float(np.linalg.norm(arr)) or 1.0
        return arr / norm


def get_embedder(cfg) -> Embedder:
    if cfg.embedding.provider == "external":
        import os

        return ExternalEmbedder(
            cfg.embedding.api_base,
            os.environ.get(cfg.embedding.api_key_env, ""),
            cfg.embedding.dim,
        )
    return LocalHashEmbedder(cfg.embedding.dim)
