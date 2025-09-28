import asyncio, json, base64, wave, io
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer
from settings import MODEL_DIR, MODEL_SUBDIR, SAMPLE_RATE, VOCAB_PATH, CHUNK_MS, MAX_CONNECTIONS
from download_model import ensure_model

# change: add allowed origins if needed
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = ensure_model()
MODEL = Model(MODEL_PATH)

def load_vocab(path: str):
    toks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        t = line.strip().lower()
        if t:
            toks.append(t)
    return toks

VOCAB = load_vocab(VOCAB_PATH)

def recognizer():
    return KaldiRecognizer(MODEL, SAMPLE_RATE, json.dumps(VOCAB))

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_SUBDIR, "sample_rate": SAMPLE_RATE, "vocab_size": len(VOCAB)}

@app.get("/vocab")
async def vocab():
    return {"vocab": VOCAB}

@app.post("/recognize")
async def recognize(file: UploadFile):
    data = await file.read()
    # expects WAV PCM 16k mono
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE:
            return JSONResponse({"error": "audio must be mono and 16kHz"}, status_code=400)
        frames = w.readframes(w.getnframes())
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
