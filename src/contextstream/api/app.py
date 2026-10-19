"""FastAPI app: WebSocket streaming chat + ingest + health + history."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from contextstream.config import load_config
from contextstream.service import RagService


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path or os.environ.get("CONTEXTSTREAM_CONFIG")
                      or Path(__file__).resolve().parents[3] / "configs" / "app.json")
    service = RagService(cfg)

    app = FastAPI(title="ContextStream", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.service = service

    class IngestRequest(BaseModel):
        id: str = Field(min_length=1)
        text: str = Field(min_length=1)

    class IngestBatch(BaseModel):
        documents: list[IngestRequest]

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "chunks_indexed": service.store.count(),
            "embedding_provider": cfg.embedding.provider,
            "llm_provider": cfg.llm.provider,
            "vector_backend": cfg.vector_store.backend,
        }

    @app.post("/ingest")
    def ingest(req: IngestBatch) -> dict:
        chunks = sum(service.ingest(d.id, d.text) for d in req.documents)
        return {"chunks_indexed": chunks, "total": service.store.count()}

    @app.get("/sessions/{session_id}/history")
    def history(session_id: str) -> dict:
        session = service.get_or_create_session(session_id)
        return {"session_id": session_id, "history": session.history}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        session = service.get_or_create_session(None)
        await ws.send_json({"type": "connected", "session_id": session.session_id})
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                kind = msg.get("type", "message")

                if kind == "resume":
                    session = service.get_or_create_session(msg.get("session_id"))
                    await ws.send_json({"type": "resumed",
                                        "session_id": session.session_id,
                                        "history": session.history})
                    continue
                if kind == "ping":
                    await ws.send_json({"type": "pong"})
                    continue
                if kind != "message":
                    await ws.send_json({"type": "error", "detail": f"unknown type {kind}"})
                    continue

                query = str(msg.get("text", "")).strip()
                if not query:
                    await ws.send_json({"type": "error", "detail": "empty message"})
                    continue
                service.record(session, "user", query)
                await ws.send_json({"type": "start", "query": query})
                final = None
                async for kind_out, payload in service.answer(query):
                    if kind_out == "delta":
                        await ws.send_json({"type": "delta", "content": payload})
                    else:
                        final = payload
                service.record(session, "assistant", final["answer"], final["sources"])
                await ws.send_json({"type": "final", **final})
        except WebSocketDisconnect:
            return

    return app


app = create_app()
