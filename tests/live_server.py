"""Live-server helper: real uvicorn subprocess + tiny sync WS client.

stdout/stderr go to a temp log file (never PIPE — an undrained pipe on a
chatty child deadlocks the boot check). Shared by scripts/smoke_test.py and
scripts/latency_test.py.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVE = PROJECT_ROOT / "scripts" / "serve.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    def __init__(self):
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self._log_fd, self._log_name = tempfile.mkstemp(prefix="contextstream-live-", suffix=".log")

    def _log_tail(self, limit: int = 2500) -> str:
        try:
            with open(self._log_name, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()[-limit:]
        except OSError:
            return "(log unavailable)"

    def __enter__(self) -> "LiveServer":
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVE), "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=str(PROJECT_ROOT), stdout=self._log_fd, stderr=self._log_fd,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self._cleanup()
                raise RuntimeError(f"server died at boot:\n{self._log_tail()}")
            try:
                if httpx.get(f"{self.base_url}/health", timeout=1.0).status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.2)
        self._cleanup()
        raise TimeoutError(f"server not healthy in 30s. Log tail:\n{self._log_tail()}")

    def _cleanup(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __exit__(self, *exc) -> None:
        self._cleanup()
        os.close(self._log_fd)
        try:
            os.unlink(self._log_name)
        except OSError:
            pass

    def websocket_connect(self, session_id: str | None = None) -> "_WsClient":
        return _WsClient(self.port, session_id)


class _WsClient:
    """Minimal sync WebSocket client over the websockets sync API."""

    def __init__(self, port: int, session_id: str | None = None):
        import websockets.sync.client

        self._ws = websockets.sync.client.connect(f"ws://127.0.0.1:{port}/ws")
        hello = json.loads(self._ws.recv(timeout=10))
        self.session_id = session_id or hello["session_id"]

    def send(self, payload: dict) -> None:
        self._ws.send(json.dumps(payload))

    def recv(self, timeout: float = 30) -> dict:
        return json.loads(self._ws.recv(timeout=timeout))

    def chat(self, text: str) -> dict:
        self.send({"type": "message", "text": text})
        final = None
        while True:
            frame = self.recv()
            if frame["type"] == "final":
                final = frame
                break
        return final

    def resume(self) -> dict:
        self.send({"type": "resume", "session_id": self.session_id})
        return self.recv()

    def close(self) -> None:
        self._ws.close()
