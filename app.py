import asyncio, json, base64, wave, io, threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Body, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer

from settings import MODEL_DIR, MODEL_SUBDIR, SAMPLE_RATE, VOCAB_PATH, CHUNK_MS
from download_model import ensure_model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change if you want to restrict origins
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = ensure_model()
MODEL = Model(MODEL_PATH)

def load_vocab(path: str) -> List[str]:
    if not Path(path).exists():
        return []
    toks: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        t = line.strip().lower()
        if t:
            toks.append(t)
    return toks

VOCAB = load_vocab(VOCAB_PATH)
VOCAB_SET = set(VOCAB)

def recognizer() -> KaldiRecognizer:
    # grammar-constrained decoding
    return KaldiRecognizer(MODEL, SAMPLE_RATE, json.dumps(VOCAB if VOCAB else []))

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_SUBDIR,
        "sample_rate": SAMPLE_RATE,
        "vocab_size": len(VOCAB),
    }

@app.get("/vocab")
async def vocab():
    return {"vocab": VOCAB}

@app.post("/recognize")
async def recognize(file: UploadFile):
    # expects WAV PCM mono at SAMPLE_RATE
    data = await file.read()
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE:
                return JSONResponse({"error": "audio must be mono and 16kHz"}, status_code=400)
            frames = w.readframes(w.getnframes())
    except wave.Error:
        return JSONResponse({"error": "invalid WAV"}, status_code=400)

    rec = recognizer()
    rec.AcceptWaveform(frames)
    res = json.loads(rec.Result())
    return {"text": res.get("text", "")}

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    rec = recognizer()
    last_partial = ""
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                pcm = msg["bytes"]
            else:
                txt = msg.get("text")
                if not txt:
                    continue
                try:
                    obj = json.loads(txt)
                    if "b64" in obj:
                        pcm = base64.b64decode(obj["b64"])
                    else:
                        continue
                except json.JSONDecodeError:
                    continue

            if rec.AcceptWaveform(pcm):
                r = json.loads(rec.Result())
                token = r.get("text", "")
                await ws.send_text(json.dumps({"type": "final", "text": token}))
                last_partial = ""
            else:
                pr = json.loads(rec.PartialResult()).get("partial", "")
                if pr and pr != last_partial:
                    await ws.send_text(json.dumps({"type": "partial", "text": pr}))
                    last_partial = pr
    except WebSocketDisconnect:
        return

# --- Token queue for Roblox HTTP polling ---
TOKEN_QUEUE: Dict[int, List[str]] = defaultdict(list)
LOCK = threading.Lock()

def _canon(s: str) -> str:
    return s.strip().lower()

@app.post("/push")
async def push_token(payload: dict = Body(...)):
    """
    JSON:
      {"uid": 123456, "text": "rot"}
      or
      {"uid": 123456, "tokens": ["rot","blau"]}
    """
    uid = int(payload["uid"])
    items: List[str] = []
    if "tokens" in payload and isinstance(payload["tokens"], list):
        items = [str(x) for x in payload["tokens"]]
    elif "text" in payload:
        items = [str(payload["text"])]
    if not items:
        return {"ok": True, "queued": 0}

    clean: List[str] = []
    for t in items:
        c = _canon(t)
        if not c:
            continue
        # filter to vocab if present
        if (not VOCAB_SET) or (c in VOCAB_SET):
            clean.append(c)

    if not clean:
        return {"ok": True, "queued": 0}

    with LOCK:
        TOKEN_QUEUE[uid].extend(clean)
        qlen = len(TOKEN_QUEUE[uid])
    return {"ok": True, "queued": qlen}

@app.get("/pull")
async def pull_tokens(uid: int = Query(...)):
    """
    Roblox polls:
      GET /pull?uid=123456 -> {"tokens": ["rot","blau"]}
    """
    with LOCK:
        lst = TOKEN_QUEUE.get(uid, [])
        TOKEN_QUEUE[uid] = []
    return {"tokens": lst}
# --- end token queue ---
