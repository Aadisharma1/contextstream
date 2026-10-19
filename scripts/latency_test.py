"""Automated latency measurement: N WS queries against a live server.

Run:  python scripts/latency_test.py [--n 20]
Writes outputs/latency.json and prints measured round-trip times.
No latency target is claimed — numbers below are what THIS machine produced.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tests.live_server import LiveServer

DOCS = [
    {"id": "lat-1", "text": "Latency measurement uses end-to-end WebSocket round trips. "
                            "A query is sent and the final streaming frame is awaited. "
                            "Measurements include retrieval and mock generation time."},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    with LiveServer() as srv:
        import httpx

        httpx.post(f"{srv.base_url}/ingest", json={"documents": DOCS}, timeout=30.0)
        queries = [f"what does measurement include? variant {i}" for i in range(args.n)]
        latencies = []
        with srv.websocket_connect() as ws:
            for q in queries:
                t0 = time.monotonic()
                ws.send({"type": "message", "text": q})
                frame = ws.recv()
                while frame["type"] != "final":
                    frame = ws.recv()
                latencies.append(round((time.monotonic() - t0) * 1000, 1))

    result = {
        "n_queries": args.n,
        "latency_ms": {
            "min": min(latencies),
            "p50": statistics.median(latencies),
            "mean": round(statistics.mean(latencies), 1),
            "p95": sorted(latencies)[int(0.95 * len(latencies)) - 1],
            "max": max(latencies),
        },
        "raw": latencies,
        "measured_on": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = Path(__file__).resolve().parents[1] / "outputs" / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "raw"}, indent=2))
    print(f"\nwritten: {out}")
    print("NOTE: these are MEASURED local numbers; no sub-500ms target is claimed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
