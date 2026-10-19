"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Source = { id: string; doc_id: string; score: number };
type HistoryEntry = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  latency_ms?: number;
};
type ConnState = "connecting" | "online" | "offline";

const WS_URL =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_WS_URL) ||
  "ws://127.0.0.1:8200/ws";

export default function Page() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [input, setInput] = useState("");
  const [conn, setConn] = useState<ConnState>("connecting");
  const [latency, setLatency] = useState<number | null>(null);
  const [streaming, setStreaming] = useState("");
  const sessionIdRef = useRef<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(500);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const connect = useCallback(() => {
    setConn("connecting");
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      backoffRef.current = 500;
      setConn("online");
    };

    ws.onmessage = (event) => {
      const frame = JSON.parse(event.data as string) as Record<string, unknown>;
      switch (frame.type) {
        case "connected":
          sessionIdRef.current = frame.session_id as string;
          break;
        case "start":
          setStreaming("");
          break;
        case "delta":
          setStreaming((prev) => prev + (frame.content as string));
          break;
        case "final": {
          const answer = frame.answer as string;
          const sources = frame.sources as Source[];
          setLatency(frame.latency_ms as number);
          setHistory((h) => [
            ...h,
            { role: "assistant", content: answer, sources, latency_ms: frame.latency_ms as number },
          ]);
          setStreaming("");
          break;
        }
        case "resumed":
          sessionIdRef.current = frame.session_id as string;
          setHistory((frame.history as HistoryEntry[]) ?? []);
          setConn("online");
          break;
        case "error":
          console.error("server error:", frame.detail);
          break;
      }
    };

    ws.onclose = () => {
      setConn("offline");
      const delay = backoffRef.current;
      backoffRef.current = Math.min(backoffRef.current * 2, 8000);
      setTimeout(() => {
        connect();
        const ws2 = wsRef.current;
        if (ws2 && sessionIdRef.current) {
          const openHandler = () => {
            ws2.send(JSON.stringify({ type: "resume", session_id: sessionIdRef.current }));
            ws2.removeEventListener("open", openHandler);
          };
          ws2.addEventListener("open", openHandler);
        }
      }, delay);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, streaming]);

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setHistory((h) => [...h, { role: "user", content: text }]);
    setInput("");
    wsRef.current.send(JSON.stringify({ type: "message", text }));
  }, [input]);

  const badge =
    conn === "online" ? ["ok", "connected"] :
    conn === "connecting" ? ["warn", "connecting"] : ["bad", "disconnected — reconnecting"];

  return (
    <main className="container">
      <h1>ContextStream</h1>
      <div className="status">
        websocket status
        <span className={`badge ${badge[0]}`}>{badge[1]}</span>
        {latency !== null && <span> · last answer {latency} ms</span>}
      </div>

      <div className="messages" aria-live="polite">
        {history.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.content}
            {m.sources && m.sources.length > 0 && (
              <div className="sources">
                sources:{" "}
                {m.sources.map((s) => `${s.id} (${s.score.toFixed(2)})`).join(", ")}
              </div>
            )}
            {m.latency_ms !== undefined && <div className="latency">{m.latency_ms} ms</div>}
          </div>
        ))}
        {streaming && (
          <div className="msg assistant">
            {streaming}
            <span className="latency">streaming…</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder={conn === "online" ? "Ask about the indexed documents…" : "waiting for connection…"}
          disabled={conn !== "online"}
        />
        <button onClick={send} disabled={conn !== "online" || !input.trim()}>
          Send
        </button>
      </div>
    </main>
  );
}
