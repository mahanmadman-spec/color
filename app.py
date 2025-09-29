# app.py
# FastAPI push–pull bridge for Roblox Color Game.
# - GET  /pull?code=XYZ            -> {"tokens": ["rot","blau",...]}  (empties the queue)
# - GET  /pull?uid=1192628416      -> same, using numeric uid instead of code
# - POST /push {code|uid, token(s)}-> queues one or more tokens for that player
# Accepts JSON or form-encoded bodies. No Vosk dependency. No external vocab file.

from __future__ import annotations

import os
import asyncio
from typing import Deque, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from collections import deque

# ----------------------------- Config ---------------------------------

# If STRICT_VOCAB=true, reject tokens not in VOCAB.
STRICT_VOCAB = os.getenv("STRICT_VOCAB", "false").lower() in ("1", "true", "yes")

# Built-in German color set (server will not transform tokens; Roblox does its own normalization).
VOCAB: set = {
    "rot","blau","grün","gelb","orange","lila","rosa","pink","braun","grau",
    "schwarz","weiß","türkis","cyan","magenta","beige","silber","gold",
    "hellblau","dunkelblau","hellgrün","dunkelgrün","dunkelrot","oliv","mint","violett",
}

# Max tokens stored per id to avoid unbounded memory.
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "64"))

# ----------------------------------------------------------------------

app = FastAPI(title="ColorGame Speech Bridge", version="1.0.0")

# Allow Roblox / Studio / Render health checks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# In-memory queues keyed by a player identifier string (code or uid)
_queues: Dict[str, Deque[str]] = {}
_lock = asyncio.Lock()  # protect _queues mutations


def _id_from_inputs(code: Optional[str], uid: Optional[Union[str, int]]) -> Optional[str]:
    if code and str(code).strip():
        return f"code:{str(code).strip()}"
    if uid is not None and str(uid).strip():
        return f"uid:{str(uid).strip()}"
    return None


def _clip_enqueue(q: Deque[str], tokens: List[str]) -> int:
    pushed = 0
    for t in tokens:
        if STRICT_VOCAB and t not in VOCAB:
            continue
        q.append(t)
        pushed += 1
        # Hard clip oldest if we exceed MAX_QUEUE
        while len(q) > MAX_QUEUE:
            q.popleft()
    return pushed


# ----------------------------- Routes ---------------------------------

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font:14px system-ui,Segoe UI,Arial;margin:40px;max-width:820px}"
        "code{background:#f4f4f7;padding:2px 4px;border-radius:4px}</style>"
        "<h2>ColorGame Speech Bridge</h2>"
        "<p>Use <code>POST /push</code> to queue tokens and <code>GET /pull?code=XYZ</code> to fetch them.</p>"
        "<ul>"
        "<li><b>POST</b> <code>/push</code> JSON: "
        "<code>{\"code\":\"ABC123\",\"token\":\"rot\"}</code> or "
        "<code>{\"code\":\"ABC123\",\"tokens\":[\"rot\",\"blau\"]}</code></li>"
        "<li><b>GET</b> <code>/pull?code=ABC123</code> → <code>{\"tokens\":[...]}</code></li>"
        "<li>Alternative identifier: <code>uid=&lt;number&gt;</code></li>"
        "</ul>"
        f"<p>STRICT_VOCAB={str(STRICT_VOCAB).lower()}, MAX_QUEUE={MAX_QUEUE}</p>"
    )


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/pull")
async def pull(code: Optional[str] = None, uid: Optional[str] = None) -> JSONResponse:
    ident = _id_from_inputs(code, uid)
    if not ident:
        return JSONResponse({"error": "missing 'code' or 'uid'"}, status_code=400)

    async with _lock:
        q = _queues.get(ident)
        if not q:
            _queues[ident] = deque()
            return JSONResponse({"tokens": []})
        # drain queue
        out: List[str] = list(q)
        q.clear()
        return JSONResponse({"tokens": out})


@app.post("/push")
async def push(request: Request) -> JSONResponse:
    """
    Accepts JSON or form data. Examples:
      JSON: {"code":"ABC","token":"rot"}
      JSON: {"uid":1192628416, "tokens":["rot","blau"]}
      FORM: code=ABC&token=rot
    Returns: {"queued": N, "id":"code:ABC"}
    """
    # Default values
    code: Optional[str] = None
    uid: Optional[Union[str, int]] = None
    tokens_in: List[str] = []

    ctype = (request.headers.get("content-type") or "").lower()

    try:
        if "application/json" in ctype:
            payload = await request.json()
            if isinstance(payload, dict):
                code = payload.get("code")
                uid = payload.get("uid")
                # Accept both "token" and "tokens"
                if "tokens" in payload and isinstance(payload["tokens"], list):
                    tokens_in = [str(t).strip() for t in payload["tokens"] if str(t).strip()]
                elif "token" in payload and isinstance(payload["token"], str):
                    t = payload["token"].strip()
                    if t:
                        tokens_in = [t]
        elif "application/x-www-form-urlencoded" in ctype or "multipart/form-data" in ctype:
            form = await request.form()
            code = str(form.get("code") or "").strip() or None
            uid_val = form.get("uid")
            uid = str(uid_val).strip() if uid_val is not None else None
            # token(s)
            token_single = form.get("token")
            if token_single:
                t = str(token_single).strip()
                if t:
                    tokens_in.append(t)
            tokens_field = form.getlist("tokens") if hasattr(form, "getlist") else None
            if tokens_field:
                for t in tokens_field:
                    s = str(t).strip()
                    if s:
                        tokens_in.append(s)
        else:
            # Attempt to parse JSON anyway; if it fails, 415
            try:
                payload = await request.json()
                if isinstance(payload, dict):
                    code = payload.get("code")
                    uid = payload.get("uid")
                    if "tokens" in payload and isinstance(payload["tokens"], list):
                        tokens_in = [str(t).strip() for t in payload["tokens"] if str(t).strip()]
                    elif "token" in payload and isinstance(payload["token"], str):
                        t = payload["token"].strip()
                        if t:
                            tokens_in = [t]
                else:
                    return JSONResponse({"error": "unsupported body"}, status_code=415)
            except Exception:
                return JSONResponse({"error": "unsupported content-type"}, status_code=415)
    except Exception as e:
        return JSONResponse({"error": f"bad request: {e}"}, status_code=400)

    ident = _id_from_inputs(code, uid)
    if not ident:
        return JSONResponse({"error": "missing 'code' or 'uid'"}, status_code=400)
    if not tokens_in:
        return JSONResponse({"error": "missing 'token' or 'tokens'"}, status_code=400)

    # Clean tokens: keep as-is; lowercasing is safe for German here, but Roblox does the canonicalization.
    tokens = [t for t in (s.strip() for s in tokens_in) if t]

    async with _lock:
        q = _queues.get(ident)
        if q is None:
            q = deque()
            _queues[ident] = q
        queued = _clip_enqueue(q, tokens)

    return JSONResponse({"queued": queued, "id": ident})


# Optional: simple text endpoint for manual testing in a browser:
#   /push-test?code=ABC&token=rot
@app.get("/push-test")
async def push_test(code: Optional[str] = None, uid: Optional[str] = None, token: Optional[str] = None):
    ident = _id_from_inputs(code, uid)
    if not ident or not token:
        return JSONResponse({"error": "provide code|uid and token"}, status_code=400)
    async with _lock:
        q = _queues.get(ident)
        if q is None:
            q = deque()
            _queues[ident] = q
        queued = _clip_enqueue(q, [token.strip()])
    return JSONResponse({"queued": queued, "id": ident})


# For local dev. Render runs: uvicorn app:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
