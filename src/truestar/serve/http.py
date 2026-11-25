"""OpenAI compatible HTTP subset of `truestar serve`, standard library only.

    POST /v1/completions        prompt (text or token ids), max_tokens, temperature, top_p, seed, stop,
                                logprobs (top k), stream
    POST /v1/chat/completions   messages, max_tokens (or max_completion_tokens), temperature, top_p, seed,
                                stop, logprobs, top_logprobs, stream
    GET  /v1/models             the one model
    GET  /health

Tokens are UTF-8 bytes (ids 0..255; the model's other ids print as <|id|>). A parameter the server does
not implement is refused by name with TS1404 when it asks for something other than the default
(n > 1, echo, penalties, logit_bias, ...), never silently ignored. A sampled request without a seed
gets one, returned in the response (TS1405), so it can be replayed exactly.
"""

from __future__ import annotations

import json
import os
import queue
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

from .engine import Engine, Request, ServeError, encode, text_of

# parameters accepted only at their default (value -> allowed defaults)
_INERT = {"n": (1, None), "best_of": (1, None), "echo": (False, None), "presence_penalty": (0, 0.0, None),
          "frequency_penalty": (0, 0.0, None), "logit_bias": (None, {}), "suffix": (None, ""),
          "user": None, "stream_options": None, "response_format": None, "parallel_tool_calls": None,
          "tools": (None, []), "tool_choice": (None, "none")}


def _check_inert(body: Dict, known: Tuple[str, ...]) -> None:
    for k, v in body.items():
        if k in known:
            continue
        if k in _INERT:
            ok = _INERT[k]
            if ok is None or v in ok:
                continue
            raise ServeError("TS1404", f"parameter {k}={v!r} is not supported by truestar serve (only its default)")
        raise ServeError("TS1404", f"unknown parameter {k!r}")


def _stops(body) -> Tuple[bytes, ...]:
    s = body.get("stop")
    if s is None:
        return ()
    s = [s] if isinstance(s, str) else list(s)
    if not all(isinstance(x, str) and x for x in s) or len(s) > 4:
        raise ServeError("TS1404", "stop must be a non empty string or a list of up to 4")
    return tuple(x.encode("utf-8") for x in s)


def _sampling(body) -> Tuple[float, float, int, bool]:
    t = float(body.get("temperature", 1.0) if body.get("temperature") is not None else 1.0)
    p = float(body.get("top_p", 1.0) if body.get("top_p") is not None else 1.0)
    seed = body.get("seed")
    assigned = False
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little") >> 1
        assigned = t > 0
    if not isinstance(seed, int):
        raise ServeError("TS1404", "seed must be an integer")
    return t, p, seed, assigned


def chat_prompt(messages) -> List[int]:
    """A plain template: <|role|> newline content newline per message, then <|assistant|> newline."""
    if not isinstance(messages, list) or not messages:
        raise ServeError("TS1404", "messages must be a non empty list")
    text = ""
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str) or "role" not in m:
            raise ServeError("TS1404", "each message needs a role and a text content")
        text += f"<|{m['role']}|>\n{m['content']}\n"
    return encode(text + "<|assistant|>\n")


def _trim(seq, stops: Tuple[bytes, ...]) -> bytes:
    b = text_of(seq.out)
    for s in stops:
        if b.endswith(s):
            return b[:-len(s)]
    return b


class Server:
    def __init__(self, engine: Engine, model_name: str = "truestar-tiny"):
        self.engine = engine
        self.model_name = model_name
        self.fingerprint = f"truestar-{engine.runner.executor}-deterministic"

    # -- request parsing
    def completion_request(self, body) -> Tuple[Request, Dict]:
        _check_inert(body, ("model", "prompt", "max_tokens", "temperature", "top_p", "seed", "stop", "logprobs",
                            "stream"))
        p = body.get("prompt")
        if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (str, list)):
            p = p[0]
        if isinstance(p, str):
            toks = encode(p)
        elif isinstance(p, list) and all(isinstance(t, int) for t in p):
            toks = list(p)
        else:
            raise ServeError("TS1404", "prompt must be one string or one list of token ids")
        t, tp, seed, assigned = _sampling(body)
        lp = body.get("logprobs")
        if lp is not None and (not isinstance(lp, int) or not 0 <= lp <= 20):
            raise ServeError("TS1404", "logprobs must be an integer 0..20")
        r = Request(prompt=toks, max_tokens=int(body.get("max_tokens") or 16), temperature=t, top_p=tp, seed=seed,
                    stop_bytes=_stops(body), top_logprobs=lp or 0)
        return r, {"seed_assigned": assigned, "logprobs": lp is not None, "stream": bool(body.get("stream"))}

    def chat_request(self, body) -> Tuple[Request, Dict]:
        _check_inert(body, ("model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p",
                            "seed", "stop", "logprobs", "top_logprobs", "stream"))
        t, tp, seed, assigned = _sampling(body)
        k = body.get("top_logprobs") or 0
        if not isinstance(k, int) or not 0 <= k <= 20:
            raise ServeError("TS1404", "top_logprobs must be an integer 0..20")
        mt = body.get("max_completion_tokens") or body.get("max_tokens") or 16
        r = Request(prompt=chat_prompt(body.get("messages")), max_tokens=int(mt), temperature=t, top_p=tp,
                    seed=seed, stop_bytes=_stops(body), top_logprobs=k)
        return r, {"seed_assigned": assigned, "logprobs": bool(body.get("logprobs")), "stream": bool(body.get("stream"))}

    # -- responses
    def _meta(self, r: Request, info: Dict) -> Dict:
        m = {"seed": r.seed}
        if info["seed_assigned"]:
            m["codes"] = ["TS1405"]
        return m

    def completion_body(self, seq, info) -> Dict:
        r = seq.req
        text = _trim(seq, r.stop_bytes)
        lp = None
        if info["logprobs"]:
            toks = [text_of([t]).decode("utf-8", "replace") for t in seq.out]
            offs, o = [], 0
            for tk in toks:
                offs.append(o)
                o += len(tk)
            lp = {"tokens": toks, "token_logprobs": list(seq.logprobs), "text_offset": offs,
                  "top_logprobs": [{text_of([k]).decode("utf-8", "replace"): v for k, v in row} for row in seq.top]
                  if r.top_logprobs else None,
                  "token_ids": list(seq.out)}
        return {"id": r.rid, "object": "text_completion", "created": int(time.time()), "model": self.model_name,
                "system_fingerprint": self.fingerprint,
                "choices": [{"index": 0, "text": text.decode("utf-8", "replace"), "logprobs": lp,
                             "finish_reason": seq.finish}],
                "usage": self._usage(seq), "truestar": self._meta(r, info)}

    def chat_body(self, seq, info) -> Dict:
        r = seq.req
        text = _trim(seq, r.stop_bytes)
        lp = None
        if info["logprobs"]:
            lp = {"content": [{"token": text_of([t]).decode("utf-8", "replace"), "token_id": t, "logprob": l,
                               "bytes": list(text_of([t])),
                               "top_logprobs": [{"token": text_of([k]).decode("utf-8", "replace"), "logprob": v,
                                                 "bytes": list(text_of([k]))} for k, v in row]}
                              for t, l, row in zip(seq.out, seq.logprobs, seq.top or [[]] * len(seq.out))]}
        return {"id": r.rid.replace("cmpl", "chatcmpl"), "object": "chat.completion", "created": int(time.time()),
                "model": self.model_name, "system_fingerprint": self.fingerprint,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text.decode("utf-8", "replace")},
                             "logprobs": lp, "finish_reason": seq.finish}],
                "usage": self._usage(seq), "truestar": self._meta(r, info)}

    @staticmethod
    def _usage(seq) -> Dict:
        return {"prompt_tokens": len(seq.req.prompt), "completion_tokens": len(seq.out),
                "total_tokens": len(seq.req.prompt) + len(seq.out),
                "prompt_tokens_details": {"cached_tokens": seq.cached}}


def make_handler(server: Server):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):   # quiet by default
            pass

        def _send(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, err: ServeError) -> None:
            self._send(code, {"error": {"message": str(err), "type": "invalid_request_error",
                                        "code": getattr(err, "code", "TS1404")}})

        def do_GET(self):
            if self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": server.model_name, "object": "model",
                                                             "owned_by": "truestar"}]})
            elif self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": {"message": f"no route {self.path}", "type": "not_found"}})

        def do_POST(self):
            chat = self.path == "/v1/chat/completions"
            if self.path not in ("/v1/completions", "/v1/chat/completions"):
                self._send(404, {"error": {"message": f"no route {self.path}", "type": "not_found"}})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ServeError("TS1404", "the body must be a JSON object")
                req, info = (server.chat_request if chat else server.completion_request)(body)
                if info["stream"]:
                    events: "queue.Queue" = queue.Queue()
                    seq = server.engine.submit_with(req, lambda s, tok: events.put(tok))
                    self._stream(seq, events, info, chat)
                    return
                seq = server.engine.submit(req)
                seq.done.wait()
                self._send(200, (server.chat_body if chat else server.completion_body)(seq, info))
            except ServeError as e:
                self._error(400, e)
            except (ValueError, TypeError) as e:
                self._error(400, ServeError("TS1404", f"bad request: {e}"))

        def _stream(self, seq, events, info, chat) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            r = seq.req
            sent = 0
            obj = "chat.completion.chunk" if chat else "text_completion"
            while True:
                tok = events.get()
                if tok is None:
                    break
                sent += 1
                piece = text_of([tok]).decode("utf-8", "replace")
                if seq.finish == "stop" and r.stop_bytes and any(text_of(seq.out[:sent]).endswith(s) for s in r.stop_bytes):
                    piece = ""
                if chat:
                    ch = {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                else:
                    ch = {"index": 0, "text": piece, "finish_reason": None,
                          "logprobs": {"tokens": [piece], "token_logprobs": [seq.logprobs[sent - 1]],
                                       "token_ids": [tok]} if info["logprobs"] else None}
                self._event({"id": r.rid, "object": obj, "model": server.model_name, "choices": [ch]})
            end = {"index": 0, "finish_reason": seq.finish}
            end.update({"delta": {}} if chat else {"text": ""})
            self._event({"id": r.rid, "object": obj, "model": server.model_name, "choices": [end],
                         "usage": server._usage(seq), "truestar": server._meta(r, info)})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _event(self, obj) -> None:
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8000, model_name: str = "truestar-tiny",
          ready=None) -> ThreadingHTTPServer:
    """Start the engine loop and return the HTTP server (call ``serve_forever`` on it)."""
    srv = ThreadingHTTPServer((host, port), make_handler(Server(engine, model_name)))
    srv.daemon_threads = True
    engine.start()
    if ready:
        ready(srv)
    return srv
