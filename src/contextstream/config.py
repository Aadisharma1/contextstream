"""Runtime configuration (JSON + env; never hardcoded secrets)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EmbeddingCfg:
    provider: str = "local"
    dim: int = 256
    api_base: str = ""
    api_key_env: str = "CONTEXTSTREAM_API_KEY"


@dataclass
class StoreCfg:
    backend: str = "memory"
    path: str = "outputs/vector_store.json"


@dataclass
class LlmCfg:
    provider: str = "mock"
    stream_delay_ms: int = 5
    model: str = "local-mock"


@dataclass
class RetrievalCfg:
    chunk_max_chars: int = 500
    chunk_overlap: int = 80
    top_k: int = 3


@dataclass
class Config:
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    vector_store: StoreCfg = field(default_factory=StoreCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)

    def resolve(self) -> "Config":
        """Apply env overrides (secrets only ever from env)."""
        if os.environ.get("CONTEXTSTREAM_EMBEDDING_PROVIDER"):
            self.embedding.provider = os.environ["CONTEXTSTREAM_EMBEDDING_PROVIDER"]
        if os.environ.get("CONTEXTSTREAM_API_BASE"):
            self.embedding.api_base = os.environ["CONTEXTSTREAM_API_BASE"]
        if os.environ.get("CONTEXTSTREAM_LLM_PROVIDER"):
            self.llm.provider = os.environ["CONTEXTSTREAM_LLM_PROVIDER"]
        if os.environ.get("CONTEXTSTREAM_VECTOR_BACKEND"):
            self.vector_store.backend = os.environ["CONTEXTSTREAM_VECTOR_BACKEND"]
        return self


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path and Path(path).exists():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for section, cls in (("embedding", EmbeddingCfg), ("vector_store", StoreCfg),
                             ("llm", LlmCfg), ("retrieval", RetrievalCfg)):
            if section in raw:
                payload = raw[section]
                unknown = set(payload) - set(cls.__dataclass_fields__)
                if unknown:
                    raise ValueError(f"unknown keys in {section}: {sorted(unknown)}")
                setattr(cfg, section, cls(**payload))
    return cfg.resolve()
