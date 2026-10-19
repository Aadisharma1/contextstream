"""API + WebSocket tests via TestClient (sync)."""

import pytest
from fastapi.testclient import TestClient

from contextstream.api.app import create_app

DOCS = [
    {"id": "zebra", "text": "Zebras are African equines with black and white stripes. "
                            "Zebras live in savannas and eat grass. Each zebra has a "
                            "unique stripe pattern like a fingerprint."},
    {"id": "quantum", "text": "Quantum computing uses qubits and superposition. "
                              "Quantum gates manipulate qubit states. Quantum computers "
                              "promise speedups for specific problems."},
]


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["chunks_indexed"] == 0
    assert h["embedding_provider"] == "local"


def test_cors_allows_dashboard_origin(client):
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"


def test_ingest_counts_chunks(client):
    r = client.post("/ingest", json={"documents": DOCS})
    assert r.status_code == 200
    body = r.json()
    assert body["chunks_indexed"] > 0
    assert body["total"] == client.get("/health").json()["chunks_indexed"]


def test_ingest_validation(client):
    r = client.post("/ingest", json={"documents": [{"id": "", "text": "x"}]})
    assert r.status_code == 422


def test_ws_roundtrip_streaming_sources_and_history(client):
    client.post("/ingest", json={"documents": DOCS})
    with client.websocket_connect("/ws") as ws:
        connected = ws.receive_json()
        assert connected["type"] == "connected"
        session_id = connected["session_id"]

        ws.send_json({"type": "message", "text": "Where do zebras live?"})
        start = ws.receive_json()
        assert start["type"] == "start"

        deltas = []
        while True:
            frame = ws.receive_json()
            if frame["type"] == "delta":
                deltas.append(frame["content"])
            elif frame["type"] == "final":
                final = frame
                break
        answer = "".join(deltas)
        assert final["answer"].strip() == answer.strip()
        assert final["latency_ms"] >= 0
        assert final["sources"], "sources must be cited"
        assert final["sources"][0]["doc_id"] == "zebra"
        assert "zebra" in answer.lower()

        h = client.get(f"/sessions/{session_id}/history").json()
        assert [e["role"] for e in h["history"]] == ["user", "assistant"]
        assert h["history"][1]["sources"]


def test_ws_resume_replays_history(client):
    client.post("/ingest", json={"documents": DOCS})
    with client.websocket_connect("/ws") as ws:
        session_id = ws.receive_json()["session_id"]
        ws.send_json({"type": "message", "text": "What is a qubit?"})
        while True:
            if ws.receive_json()["type"] == "final":
                break

    with client.websocket_connect("/ws") as ws2:
        ws2.receive_json()
        ws2.send_json({"type": "resume", "session_id": session_id})
        resumed = ws2.receive_json()
        assert resumed["type"] == "resumed"
        assert resumed["session_id"] == session_id
        assert len(resumed["history"]) == 2


def test_ws_ping_pong_and_errors(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
        ws.send_json({"type": "message", "text": ""})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "bogus"})
        assert ws.receive_json()["type"] == "error"
