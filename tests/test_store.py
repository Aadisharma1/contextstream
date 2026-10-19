import numpy as np
import pytest

from contextstream.rag.store import FileVectorStore, InMemoryVectorStore


def _unit(vecs):
    arr = np.asarray(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_file_store_roundtrip_across_instances(tmp_path):
    path = tmp_path / "store.json"
    s1 = FileVectorStore(path)
    s1.upsert(["a", "b"], _unit([[1, 0], [0, 1]]), [{"text": "alpha"}, {"text": "beta"}])
    assert s1.count() == 2

    s2 = FileVectorStore(path)
    assert s2.count() == 2
    hits = s2.query(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert hits[0]["id"] == "a"
    assert hits[0]["text"] == "alpha"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_file_store_upsert_replaces_existing_id(tmp_path):
    path = tmp_path / "store.json"
    s = FileVectorStore(path)
    s.upsert(["a"], _unit([[1, 0]]), [{"text": "old"}])
    s.upsert(["a"], _unit([[0, 1]]), [{"text": "new"}])
    assert s.count() == 1
    hits = FileVectorStore(path).query(np.array([0, 1.0], dtype=np.float32), top_k=1)
    assert hits[0]["text"] == "new"


def test_file_store_empty_query(tmp_path):
    s = FileVectorStore(tmp_path / "empty.json")
    assert s.count() == 0
    assert s.query(np.array([1.0, 0.0]), top_k=3) == []


def test_file_store_upsert_length_mismatch_rejected(tmp_path):
    s = FileVectorStore(tmp_path / "x.json")
    with pytest.raises(ValueError):
        s.upsert(["a", "b"], _unit([[1, 0]]), [{"text": "x"}])


def test_file_and_memory_stores_agree_on_scores(tmp_path):
    rng = np.random.default_rng(0)
    vecs = _unit(rng.normal(0, 1, size=(6, 16)))
    meta = [{"text": f"doc {i}"} for i in range(6)]
    mem = InMemoryVectorStore()
    mem.upsert([f"d{i}" for i in range(6)], vecs, meta)
    fil = FileVectorStore(tmp_path / "same.json")
    fil.upsert([f"d{i}" for i in range(6)], vecs, meta)

    q = _unit(rng.normal(0, 1, size=(1, 16)))[0]
    a = mem.query(q, top_k=3)
    b = fil.query(q, top_k=3)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    for x, y in zip(a, b):
        assert x["score"] == pytest.approx(y["score"], abs=1e-5)
