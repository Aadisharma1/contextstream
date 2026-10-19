"""Vector store abstraction: in-memory default, JSON-file persistence, Pinecone adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None: ...
    def query(self, vector: np.ndarray, top_k: int) -> list[dict]: ...
    def count(self) -> int: ...


class InMemoryVectorStore:
    """Cosine-similarity store. Persisted only for process lifetime."""

    def __init__(self):
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._metadata: list[dict] = []

    def upsert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
        if len(ids) != len(metadata) or vectors.shape[0] != len(ids):
            raise ValueError("ids/vectors/metadata length mismatch")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vectors / norms
        keep_new = list(range(len(ids)))
        if self._matrix is not None:
            existing = {id_: i for i, id_ in enumerate(self._ids)}
            for new_pos, id_ in enumerate(ids):
                if id_ in existing:
                    old_i = existing[id_]
                    self._matrix[old_i] = vecs[new_pos]
                    self._metadata[old_i] = metadata[new_pos]
                    keep_new.remove(new_pos)
        if keep_new:
            sel = np.array(keep_new)
            self._matrix = vecs[sel] if self._matrix is None else np.vstack([self._matrix, vecs[sel]])
            self._ids += [ids[i] for i in keep_new]
            self._metadata += [metadata[i] for i in keep_new]

    def query(self, vector: np.ndarray, top_k: int) -> list[dict]:
        if self._matrix is None or not self._ids:
            return []
        v = vector / (np.linalg.norm(vector) or 1.0)
        scores = self._matrix @ v
        order = np.argsort(-scores)[:top_k]
        return [{"id": self._ids[i], "score": float(scores[i]), **self._metadata[i]}
                for i in order]

    def count(self) -> int:
        return len(self._ids)


class FileVectorStore:
    """JSON-file-backed store: the index survives process restarts.

    Appropriate for demo scale (hundreds to low thousands of chunks). Rows are
    L2-normalized on upsert, so query() is exact cosine similarity, same as the
    in-memory store. Every upsert rewrites the file — batch your ingestion.
    """

    def __init__(self, path: str | Path = "outputs/vector_store.json"):
        self._path = Path(path)
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._metadata: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._ids = list(data["ids"])
            self._vectors = [list(map(float, v)) for v in data["vectors"]]
            self._metadata = list(data["metadata"])

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({
            "ids": self._ids,
            "vectors": self._vectors,
            "metadata": self._metadata,
        }), encoding="utf-8")

    def upsert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
        if len(ids) != len(metadata) or vectors.shape[0] != len(ids):
            raise ValueError("ids/vectors/metadata length mismatch")
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        index = {id_: i for i, id_ in enumerate(self._ids)}
        for pos, id_ in enumerate(ids):
            row = [float(x) for x in arr[pos]]
            if id_ in index:
                i = index[id_]
                self._vectors[i] = row
                self._metadata[i] = metadata[pos]
            else:
                index[id_] = len(self._ids)
                self._ids.append(id_)
                self._vectors.append(row)
                self._metadata.append(metadata[pos])
        self._save()

    def query(self, vector: np.ndarray, top_k: int) -> list[dict]:
        if not self._ids:
            return []
        m = np.asarray(self._vectors, dtype=np.float32)
        v = np.asarray(vector, dtype=np.float32)
        v = v / (np.linalg.norm(v) or 1.0)
        scores = m @ v
        order = np.argsort(-scores)[:top_k]
        return [{"id": self._ids[i], "score": float(scores[i]), **self._metadata[i]}
                for i in order]

    def count(self) -> int:
        return len(self._ids)


class PineconeVectorStore:
    """Pinecone adapter. Requires PINECONE_API_KEY; fails loudly otherwise.

    NOTE: untested in the authoring environment (no Pinecone account); the
    interface matches the in-memory store so the swap is config-only.
    """

    def __init__(self, index_name: str = "contextstream", dim: int = 256):
        import os

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Pinecone backend selected but PINECONE_API_KEY is not set. "
                "Either export it or set vector_store.backend='memory' (default)."
            )
        from pinecone import Pinecone

        self._client = Pinecone(api_key=api_key)
        if index_name not in self._client.list_indexes().names():
            self._client.create_index(index_name, dimension=dim, metric="cosine")
        self._index = self._client.Index(index_name)

    def upsert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
        self._index.upsert(vectors=[
            {"id": i, "values": v.tolist(), "metadata": m}
            for i, v, m in zip(ids, vectors, metadata)
        ])

    def query(self, vector: np.ndarray, top_k: int) -> list[dict]:
        res = self._index.query(vector=vector.tolist(), top_k=top_k, include_metadata=True)
        return [{"id": m["id"], "score": m["score"], **(m.get("metadata") or {})}
                for m in res["matches"]]

    def count(self) -> int:
        return int(self._index.describe_index_stats()["total_vector_count"])


def build_store(cfg):
    if cfg.vector_store.backend == "pinecone":
        return PineconeVectorStore(dim=cfg.embedding.dim)
    if cfg.vector_store.backend == "file":
        return FileVectorStore(cfg.vector_store.path)
    return InMemoryVectorStore()
