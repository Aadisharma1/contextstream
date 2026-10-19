"""End-to-end smoke test: live server, ingest, WS streaming, reconnect.

Run:  python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tests.live_server import LiveServer


def main() -> int:
    import httpx

    DOCS = [
        {"id": "kb-1", "text": "ContextStream indexes documents into overlapping chunks. "
                               "Chunks are embedded and stored in a vector database. "
                               "Retrieval selects the most similar chunks for a query."},
        {"id": "kb-2", "text": "The WebSocket protocol supports streaming responses. "
                               "Clients reconnect automatically after a dropped "
                               "connection and resume their session history."},
    ]
    with LiveServer() as srv:
        c = httpx.Client(base_url=srv.base_url, timeout=30.0)
        print("=== CONTEXTSTREAM SMOKE TEST ===")
        print(f"health: {c.get('/health').json()}")
        r = c.post("/ingest", json={"documents": DOCS})
        print(f"ingest: {r.json()}")
        assert r.status_code == 200

        with srv.websocket_connect() as ws:
            print(f"connected: session {ws.session_id[:8]}...")
            final = ws.chat("How are documents indexed?")
            print(f"answer: {final['answer'][:100]}...")
            print(f"sources: {[s['id'] for s in final['sources']]} latency: {final['latency_ms']}ms")
            assert final["sources"] and final["latency_ms"] >= 0

        with srv.websocket_connect(session_id=ws.session_id) as ws2:
            resumed = ws2.resume()
            print(f"resumed session with {len(resumed['history'])} history entries")
            assert len(resumed["history"]) == 2
        c.close()
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
