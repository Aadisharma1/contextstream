import numpy as np
import pytest

from contextstream.config import Config, EmbeddingCfg, RetrievalCfg
from contextstream.rag.chunker import chunk_text
from contextstream.rag.embeddings import LocalHashEmbedder
from contextstream.rag.store import InMemoryVectorStore
from contextstream.rag.embeddings import _tokenize


def test_chunker_sizes_overlap_and_order():
    text = " ".join(f"Sentence number {i} talks about topic {i % 5}." for i in range(40))
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert chunks, "must produce chunks"
    assert all(len(c) <= 300 for c in chunks)
    assert len(chunks) > 1
    assert chunks[1].startswith(chunks[0][-50:])
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=10, overlap=10)
    assert chunk_text("   ", 300, 50) == []
    assert chunk_text("short", 300, 50) == ["short"]


def test_embedder_deterministic_and_normalized():
    e = LocalHashEmbedder(dim=128)
    v1 = e.embed("The quick brown fox")
    v2 = e.embed("The quick brown fox")
    assert np.array_equal(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-6
    assert v1.shape == (128,)
    assert not np.array_equal(e.embed("cats"), e.embed("dogs"))
    assert np.array_equal(e.embed(""), np.zeros(128, dtype=np.float32))


def test_tokenizer_strips_punct():
    assert _tokenize("Hello, WORLD!!") == ["hello", "world"]


def test_store_cosine_topk_and_upsert_replace():
    store = InMemoryVectorStore()
    vecs = np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32)
    store.upsert(["a", "b", "c"], vecs, [{"text": f"doc {i}"} for i in range(3)])
    assert store.count() == 3
    hits = store.query(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert hits[0]["id"] == "a"
    assert hits[0]["score"] > hits[1]["score"]
    store.upsert(["a"], np.array([[0, 1]], dtype=np.float32), [{"text": "updated"}])
    assert store.count() == 3
    hits = store.query(np.array([0, 1.0], dtype=np.float32), top_k=1)
    assert hits[0]["id"] == "a" and hits[0]["score"] > 0.999
    assert hits[0]["text"] == "updated"


def test_retriever_orders_relevant_document_first():
    from contextstream.service import RagService

    cfg = Config(embedding=EmbeddingCfg(provider="local", dim=128),
                 retrieval=RetrievalCfg(chunk_max_chars=200, chunk_overlap=20, top_k=2))
    svc = RagService(cfg)
    svc.ingest("zebra_doc", "Zebras are African equines with black and white stripes. "
                           "Zebras live in savannas and grasslands.")
    svc.ingest("quantum_doc", "Quantum computing uses qubits and superposition. "
                              "Quantum gates manipulate qubit states.")
    hits = svc.retrieve("what do zebras eat in the savanna?")
    assert hits[0]["doc_id"] == "zebra_doc"
    assert hits[0]["score"] >= hits[-1]["score"]


@pytest.mark.asyncio
async def test_mock_llm_streams_full_answer():
    from contextstream.llm.providers import MockLLM

    llm = MockLLM(stream_delay_ms=0)
    chunks = [{"id": "d#0", "text": "Zebras eat grass. Zebras drink water daily."}]
    parts = [delta async for delta in llm.stream("what do zebras eat?", chunks)]
    answer = "".join(parts)
    assert "grass" in answer.lower()
    assert answer.endswith(" ")
