# app.py
# FastAPI + Vosk German STT, queue-per-code bridge, WAV-friendly recorder page.
# Robust to missing vocab, variable sample rates, and empty chunks.

import os, io, time, json, queue, threading
from typing import Dict, Set, Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from vosk import Model, KaldiRecognizer
import soundfile as sf

# ------------------------------- Config ---------------------------------------

START_TS = time.monotonic()

# Model directory (mounted in repo: models/vosk-model-small-de-0.15 by default)
MODEL_DIR = os.getenv("VOSK_MODEL_DIR", "models/vosk-model-small-de-0.15")

# Optional path to a custom vocab file (one token per line, lower-case ASCII)
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab/colors_de.txt")

DEFAULT_VOCAB: Set[str] = {
    "rot","blau","gruen","gelb","orange","lila","rosa","pink","braun","grau",
    "schwarz","weiss","tuerkis","cyan","magenta","beige","silber","gold",
    "hellblau","dunkelblau","hellgruen","dunkelgruen","dunkelrot","oliv","mint","violett"
}

# Soft constraints: single word, from the vocabulary
MAX_QUEUE_LEN_PER_CODE = 64

# ------------------------------------------------------------------------------
# Load vocabulary with fallback
def load_vocab(path: str) -> Set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            items = {s.strip().lower() for s in f.read().splitlines() if s.strip()}
            return items if items else set(DEFAULT_VOCAB)
    except FileNotFoundError:
        return set(DEFAULT_VOCAB)

VOCAB: Set[str] = load_vocab(VOCAB_PATH)

# Normalization: map common German forms to ASCII tokens used in game
ALIASES: Dict[str, str] = {
    "weiß": "weiss", "weiss": "weiss",
    "grün": "gruen", "gruen": "gruen",
    "türkis": "tuerkis", "turkis": "tuerkis", "türkys": "tuerkis",
    "hell-blau": "hellblau", "dunkel-blau": "dunkelblau",
    "hell-grün": "hellgruen", "dunkel-grün": "dunkelgruen",
    "schwarz.": "schwarz", "weiss.": "weiss",
    "violet": "violett", "purple": "violett",
}

def normalize_token(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip().lower()
    # strip non-letters except space and hyphen
    s = "".join(ch if ("a" <= ch <= "z" or ch in " äöüß-") else " " for ch in s)
    s = " ".join(s.split())
    if not s:
        return None
    # direct lookup of complete phrase
    if s in ALIASES:
        s = ALIASES[s]
    # split and try words
    parts = s.replace("-", "").split()
    for w in parts:
        w = ALIASES.get(w, w)
        if w in VOCAB:
            return w
    return None

# ------------------------------------------------------------------------------
# Load Vosk model once
try:
    MODEL = Model(MODEL_DIR)
except Exception as e:
    # Provide a clear message; Render will show this in logs
    raise RuntimeError(f"Failed to load Vosk model from '{MODEL_DIR}': {e}")

# ------------------------------------------------------------------------------
# Per-code token queues
QUEUES: Dict[str, "queue.Queue[str]"] = {}
QLEN: Dict[str, int] = {}  # approximate length tracking

def q_for(code: str) -> "queue.Queue[str]":
    code = code.strip()
    if code not in QUEUES:
        QUEUES[code] = queue.Queue()
        QLEN[code] = 0
    return QUEUES[code]

def push_token(code: str, token: str) -> None:
    q = q_for(code)
    # Fast length cap to avoid unbounded memory if client stops polling
    if QLEN.get(code, 0) >= MAX_QUEUE_LEN_PER_CODE:
        try:
            q.get_nowait()
            QLEN[code] = max(0, QLEN[code] - 1)
        except Exception:
            pass
    q.put(token)
    QLEN[code] = QLEN.get(code, 0) + 1

def drain_tokens(code: str) -> List[str]:
    out: List[str] = []
    q = q_for(code)
    try:
        while True:
            out.append(q.get_nowait())
            QLEN[code] = max(0, QLEN.get(code, 0) - 1)
    except Exception:
        pass
    return out

# ------------------------------------------------------------------------------
# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Health
@app.get("/health")
def health():
    return {
        "ok": True,
        "model": os.path.basename(MODEL_DIR),
        "vocab": len(VOCAB),
        "uptime_sec": round(time.monotonic() - START_TS, 2)
    }

# ------------------------------------------------------------------------------
# Recognize endpoint: accepts a short audio chunk and enqueues one token (if any)
@app.post("/recognize")
async def recognize(code: str = Form(...), audio: UploadFile = File(...)):
    if not code or len(code) > 64:
        raise HTTPException(status_code=400, detail="invalid code")

    raw = await audio.read()
    if not raw or len(raw) < 512:  # ignore tiny chunks
        return {"ok": True, "token": None}

    # Decode audio with soundfile (supports WAV/FLAC/OGG/AIFF via libsndfile)
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="int16", always_2d=False)
    except Exception as e:
        # Return ok but no token; front-end keeps sending next chunk
        return {"ok": True, "token": None, "note": "decode_failed"}

    # Vosk recognizer needs correct sample rate
    try:
        rec = KaldiRecognizer(MODEL, float(sr))
        # int16 ndarray -> bytes
        buf = data.tobytes()
        ok = rec.AcceptWaveform(buf)
        txt = rec.FinalResult() if ok else rec.PartialResult()
    except Exception:
        return {"ok": True, "token": None}

    # Extract candidate word
    try:
        j = json.loads(txt)
        hyp = (j.get("text") or j.get("partial") or "").strip()
    except Exception:
        hyp = ""

    token = normalize_token(hyp)
    if token:
        push_token(code, token)
        return {"ok": True, "token": token}
    return {"ok": True, "token": None}

# ------------------------------------------------------------------------------
# Pull endpoint: Roblox polls this to collect tokens for a code
@app.get("/pull")
def pull(code: str):
    if not code or len(code) > 64:
        raise HTTPException(status_code=400, detail="invalid code")
    items = drain_tokens(code)
    return {"tokens": items}

# ------------------------------------------------------------------------------
# Minimal browser bridge page:
# Records raw PCM via AudioWorklet (or ScriptProcessor fallback), packs 16-bit WAV,
# POSTs each ~1s chunk to /recognize with the same code.
@app.get("/bridge")
def bridge():
    html = f"""<!doctype html><meta charset="utf-8">
<title>Mic-Bridge</title>
<style>
body{{font:16px system-ui,Segoe UI,Arial;margin:24px;max-width:880px;background:#0b0b0f;color:#eaeaf0}}
h1{{font-size:22px;margin:0 0 14px}}
label,input,button{{font:inherit}}
input#code{{font:700 22px ui-monospace,Consolas,monospace;width:14ch;text-transform:uppercase;padding:6px 8px;border-radius:8px;border:1px solid #334}}
button{{padding:8px 14px;border-radius:10px;border:1px solid #46f;background:#9df;color:#012;font-weight:700;cursor:pointer}}
#hint{{margin-left:12px;opacity:.85}}
#log{{white-space:pre-wrap;background:#10131a;border:1px solid #233;padding:12px;border-radius:10px;margin-top:16px;min-height:120px}}
.small{{opacity:.8;font-size:13px}}
kbd{{background:#222;border:1px solid #444;padding:2px 5px;border-radius:4px}}
</style>
<h1>Mic-Bridge</h1>
<p>Enter the code you see in Roblox. Keep this tab open while you play.</p>
<p>
  <label for="code">Code:</label>
  <input id="code" placeholder="XXXXXXXX" maxlength="16">
  <button id="go">Start</button>
  <span id="hint"></span>
</p>
<div class="small">If your browser asks for microphone access, allow it.</div>
<div id="log"></div>

<script>
const base = location.origin;
const $ = s => document.querySelector(s);
const log = msg => {{ const d = $("#log"); d.textContent = msg + "\\n" + d.textContent; }};

let ctx, node, media, running = false, code = "", sr = 48000;
const CHUNK_MS = 1000;

function pcm16leFromFloat32(f32){
  const out = new Int16Array(f32.length);
  for (let i=0; i<f32.length; i++){ let s = Math.max(-1, Math.min(1, f32[i])); out[i] = (s < 0 ? s * 0x8000 : s * 0x7FFF)|0; }
  return out;
}

function wavEncode(int16, sampleRate){
  const numFrames = int16.length;
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numFrames * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);
  let p = 0;
  function w4(s){ for(let i=0;i<4;i++) view.setUint8(p++, s.charCodeAt(i)); }
  function w2(v){ view.setUint16(p, v, true); p+=2; }
  function w4i(v){ view.setUint32(p, v, true); p+=4; }
  w4('RIFF'); w4i(36 + dataSize); w4('WAVE');
  w4('fmt '); w4i(16); w2(1); w2(numChannels); w4i(sampleRate); w4i(byteRate); w2(blockAlign); w2(16);
  w4('data'); w4i(dataSize);
  const out = new Int16Array(buf, 44, numFrames);
  out.set(int16);
  return new Blob([buf], {{type:'audio/wav'}});
}

async function postChunk(f32){
  const i16 = pcm16leFromFloat32(f32);
  const wav = wavEncode(i16, sr);
  const fd  = new FormData();
  fd.append('code', code);
  fd.append('audio', wav, 'chunk.wav');
  try {{
    const r = await fetch(base + '/recognize', {{method:'POST', body: fd}});
    const j = await r.json();
    if (j && j.token) log('→ ' + j.token);
  }} catch(e) {{}}
}

async function start(){
  code = ($("#code").value || "").trim();
  if (!code) {{ alert("Enter code"); return; }}
  $("#hint").textContent = "Recording… keep this tab open.";
  try {{
    media = await navigator.mediaDevices.getUserMedia({{audio: true, video: false}});
  }} catch(e) {{
    alert("Microphone permission denied."); return;
  }}

  ctx = new (window.AudioContext || window.webkitAudioContext)();
  sr = ctx.sampleRate || 48000;

  const src = ctx.createMediaStreamSource(media);

  // Worklet path (preferred)
  if (ctx.audioWorklet) {{
    const workletJS = `
      class Capture extends AudioWorkletProcessor {{
        constructor(){{
          super(); this.buf = []; this.samples = 0; this.target = sampleRate; this.chunk = Math.floor(this.target * {CHUNK_MS}/1000);
        }}
        process(inputs) {{
          const ch = inputs[0][0];
          if (!ch) return true;
          this.buf.push(new Float32Array(ch));
          this.samples += ch.length;
          if (this.samples >= this.chunk) {{
            let tot = 0; for (const b of this.buf) tot += b.length;
            const out = new Float32Array(tot);
            let o=0; for (const b of this.buf) {{ out.set(b, o); o+=b.length; }}
            this.buf = []; this.samples = 0;
            this.port.postMessage(out, [out.buffer]);
          }}
          return true;
        }}
      }}
      registerProcessor('capture', Capture);
    `;
    const blob = new Blob([workletJS], {{type:'application/javascript'}});
    await ctx.audioWorklet.addModule(URL.createObjectURL(blob));
    const aw = new AudioWorkletNode(ctx, 'capture');
    aw.port.onmessage = (ev)=> postChunk(ev.data);
    src.connect(aw).connect(ctx.destination);
    running = true;
    return;
  }}

  // Fallback: ScriptProcessor (deprecated but widely supported)
  const frame = 2048;
  const proc = ctx.createScriptProcessor(frame, 1, 1);
  let acc = [];
  let accLen = 0;
  proc.onaudioprocess = (ev)=>{
    const ch = ev.inputBuffer.getChannelData(0);
    acc.push(new Float32Array(ch)); accLen += ch.length;
    const target = Math.floor(sr * {CHUNK_MS}/1000);
    if (accLen >= target){
      let tot=0; for (const b of acc) tot += b.length;
      const out = new Float32Array(tot);
      let o=0; for (const b of acc) {{ out.set(b, o); o+=b.length; }}
      acc = []; accLen = 0;
      postChunk(out);
    }
  };
  src.connect(proc); proc.connect(ctx.destination);
  running = true;
}

document.getElementById('go').addEventListener('click', start);
</script>
"""
    return HTMLResponse(content=html)

# ------------------------------------------------------------------------------
# Root
@app.get("/")
def root():
    return PlainTextResponse("OK. See /health, /bridge, /recognize, /pull.")

# ------------------------------------------------------------------------------
# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"ok": False, "detail": exc.detail})

@app.exception_handler(Exception)
async def unknown_exception_handler(request: Request, exc: Exception):
    # avoid leaking internals; keep logs in Render dashboard
    return JSONResponse(status_code=500, content={"ok": False, "detail": "server_error"})
