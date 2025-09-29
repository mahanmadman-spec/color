# FastAPI + Vosk (German) speech bridge.
# Endpoints:
#   GET  /health         → { ok, model, vocab, uptime_sec }
#   POST /recognize      ← form(code, file=audio-blob) → { ok, token|null }
#   GET  /pull?code=XYZ  → { tokens: ["rot", ...] }
#   GET  /bridge         → simple browser recorder UI
#   GET  /               → plaintext OK

import os, io, time, json, queue, zipfile, tempfile, shutil, urllib.request
from typing import Dict, Set, Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from vosk import Model, KaldiRecognizer
import soundfile as sf

# ------------------------------- config ---------------------------------------

START_TS = time.monotonic()

# Model config (can override via Render env)
MODEL_DIR  = os.getenv("VOSK_MODEL_DIR", "models/vosk-model-small-de-0.15")
MODEL_URL  = os.getenv("VOSK_MODEL_URL", "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip")

# Vocabulary config
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab/colors_de.txt")

MAX_QUEUE_LEN_PER_CODE = 64

DEFAULT_VOCAB: Set[str] = {
    "rot","blau","gruen","gelb","orange","lila","rosa","pink","braun","grau",
    "schwarz","weiss","tuerkis","cyan","magenta","beige","silber","gold",
    "hellblau","dunkelblau","hellgruen","dunkelgruen","dunkelrot","oliv","mint","violett"
}

ALIASES: Dict[str, str] = {
    "weiß":"weiss","weiss":"weiss",
    "grün":"gruen","gruen":"gruen",
    "türkis":"tuerkis","turkis":"tuerkis",
    "hell-blau":"hellblau","dunkel-blau":"dunkelblau",
    "hell-grün":"hellgruen","dunkel-grün":"dunkelgruen",
    "violet":"violett","purple":"violett",
}

# ------------------------------- helpers --------------------------------------

def load_vocab(path: str) -> Set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            items = {s.strip().lower() for s in f.read().splitlines() if s.strip()}
            return items if items else set(DEFAULT_VOCAB)
    except FileNotFoundError:
        return set(DEFAULT_VOCAB)

VOCAB: Set[str] = load_vocab(VOCAB_PATH)

def normalize_token(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip().lower()
    s = "".join(ch if ("a" <= ch <= "z" or ch in " äöüß-") else " " for ch in s)
    s = " ".join(s.split())
    if not s:
        return None
    s = ALIASES.get(s, s)
    parts = s.replace("-", "").split()
    for w in parts:
        w = ALIASES.get(w, w)
        if w in VOCAB:
            return w
    return None

def model_ok(path: str) -> bool:
    # Vosk small models include graph/Gr.fst; this is a reliable check
    return os.path.exists(os.path.join(path, "graph", "Gr.fst"))

def ensure_model(path: str, url: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if model_ok(path):
        return
    # Download + extract
    tmp_zip = None
    try:
        print(f"[bootstrap] downloading model from {url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp_zip = tmp.name
            with urllib.request.urlopen(url, timeout=300) as r:
                shutil.copyfileobj(r, tmp)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            # Extract into models/, preserving the top-level folder
            extract_root = os.path.abspath(os.path.join(path, os.pardir))
            zf.extractall(extract_root)
        # If extracted folder name differs from expected MODEL_DIR, try to find it and rename
        if not model_ok(path):
            parent = os.path.abspath(os.path.join(path, os.pardir))
            for name in os.listdir(parent):
                candidate = os.path.join(parent, name)
                if os.path.isdir(candidate) and model_ok(candidate):
                    if os.path.abspath(candidate) != os.path.abspath(path):
                        try:
                            os.replace(candidate, path)
                        except Exception:
                            # fallback: leave as-is; app will try to init from candidate directly
                            pass
                    break
        if not model_ok(path):
            raise RuntimeError("model extracted but structure invalid")
    finally:
        if tmp_zip and os.path.exists(tmp_zip):
            try: os.remove(tmp_zip)
            except Exception: pass

# ------------------------------- model load -----------------------------------

# Attempt to ensure the model is present before loading
ensure_model(MODEL_DIR, MODEL_URL)

# Load model (raises if still invalid)
MODEL = Model(MODEL_DIR)

# ------------------------------- queues ---------------------------------------

QUEUES: Dict[str, "queue.Queue[str]"] = {}
QLEN: Dict[str, int] = {}

def q_for(code: str) -> "queue.Queue[str]":
    code = code.strip()
    if code not in QUEUES:
        QUEUES[code] = queue.Queue()
        QLEN[code] = 0
    return QUEUES[code]

def push_token(code: str, token: str) -> None:
    q = q_for(code)
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

# ------------------------------- app ------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# ------------------------------- endpoints ------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": os.path.basename(MODEL_DIR),
        "vocab": len(VOCAB),
        "uptime_sec": round(time.monotonic() - START_TS, 2),
        "model_ok": model_ok(MODEL_DIR),
    }

@app.post("/recognize")
async def recognize(code: str = Form(...), audio: UploadFile = File(...)):
    if not code or len(code) > 64:
        raise HTTPException(status_code=400, detail="invalid code")

    raw = await audio.read()
    if not raw or len(raw) < 512:
        return {"ok": True, "token": None}

    # decode with libsndfile (we send WAV from /bridge; this is robust)
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="int16", always_2d=False)
    except Exception:
        return {"ok": True, "token": None, "note": "decode_failed"}

    try:
        rec = KaldiRecognizer(MODEL, float(sr))
        ok = rec.AcceptWaveform(data.tobytes())
        txt = rec.FinalResult() if ok else rec.PartialResult()
    except Exception:
        return {"ok": True, "token": None}

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

@app.get("/pull")
def pull(code: str):
    if not code or len(code) > 64:
        raise HTTPException(status_code=400, detail="invalid code")
    return {"tokens": drain_tokens(code)}

# Plain JS/HTML (no Python f-strings here)
HTML_BRIDGE = """
<!doctype html><meta charset="utf-8">
<title>Mic-Bridge</title>
<style>
body{font:16px system-ui,Segoe UI,Arial;margin:24px;max-width:880px;background:#0b0b0f;color:#eaeaf0}
h1{font-size:22px;margin:0 0 14px}
label,input,button{font:inherit}
input#code{font:700 22px ui-monospace,Consolas,monospace;width:14ch;text-transform:uppercase;padding:6px 8px;border-radius:8px;border:1px solid #334}
button{padding:8px 14px;border-radius:10px;border:1px solid #46f;background:#9df;color:#012;font-weight:700;cursor:pointer}
#hint{margin-left:12px;opacity:.85}
#log{white-space:pre-wrap;background:#10131a;border:1px solid #233;padding:12px;border-radius:10px;margin-top:16px;min-height:120px}
.small{opacity:.8;font-size:13px}
kbd{background:#222;border:1px solid #444;padding:2px 5px;border-radius:4px}
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
const log = msg => { const d = $("#log"); d.textContent = msg + "\\n" + d.textContent; };

let ctx, media, code = "", sr = 48000;
const CHUNK_MS = 1000;

function pcm16leFromFloat32(f32){
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++){
    let s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = (s < 0 ? s * 0x8000 : s * 0x7FFF) | 0;
  }
  return out;
}

function wavEncode(int16, sampleRate){
  const numFrames = int16.length, numChannels = 1, bytesPerSample = 2;
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
  return new Blob([buf], {type:'audio/wav'});
}

async function postChunk(f32){
  const i16 = pcm16leFromFloat32(f32);
  const wav = wavEncode(i16, sr);
  const fd  = new FormData();
  fd.append('code', code);
  fd.append('audio', wav, 'chunk.wav');
  try {
    const r = await fetch(base + '/recognize', {method:'POST', body: fd});
    const j = await r.json();
    if (j && j.token) log('→ ' + j.token);
  } catch(e) {}
}

async function start(){
  code = ($("#code").value || "").trim();
  if (!code) { alert("Enter code"); return; }
  $("#hint").textContent = "Recording… keep this tab open.";
  try {
    media = await navigator.mediaDevices.getUserMedia({audio: true, video: false});
  } catch(e) {
    alert("Microphone permission denied."); return;
  }
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  sr = ctx.sampleRate || 48000;

  const src = ctx.createMediaStreamSource(media);

  if (ctx.audioWorklet) {
    const workletJS = `
      class Capture extends AudioWorkletProcessor {
        constructor(){
          super(); this.buf = []; this.samples = 0; this.target = sampleRate;
          this.chunk = Math.floor(this.target * ${CHUNK_MS}/1000);
        }
        process(inputs){
          const ch = inputs[0][0];
          if (!ch) return true;
          this.buf.push(new Float32Array(ch));
          this.samples += ch.length;
          if (this.samples >= this.chunk) {
            let tot = 0; for (const b of this.buf) tot += b.length;
            const out = new Float32Array(tot);
            let o=0; for (const b of this.buf) { out.set(b, o); o += b.length; }
            this.buf = []; this.samples = 0;
            this.port.postMessage(out, [out.buffer]);
          }
          return true;
        }
      }
      registerProcessor('capture', Capture);
    `;
    const blob = new Blob([workletJS], {type:'application/javascript'});
    await ctx.audioWorklet.addModule(URL.createObjectURL(blob));
    const aw = new AudioWorkletNode(ctx, 'capture');
    aw.port.onmessage = (ev)=> postChunk(ev.data);
    src.connect(aw).connect(ctx.destination);
    return;
  }

  // Fallback: ScriptProcessor
  const frame = 2048;
  const proc = ctx.createScriptProcessor(frame, 1, 1);
  let acc = [], accLen = 0;
  proc.onaudioprocess = (ev)=>{
    const ch = ev.inputBuffer.getChannelData(0);
    acc.push(new Float32Array(ch)); accLen += ch.length;
    const target = Math.floor(sr * CHUNK_MS / 1000);
    if (accLen >= target){
      let tot=0; for (const b of acc) tot += b.length;
      const out = new Float32Array(tot);
      let o=0; for (const b of acc) { out.set(b, o); o += b.length; }
      acc = []; accLen = 0;
      postChunk(out);
    }
  };
  src.connect(proc); proc.connect(ctx.destination);
}

document.getElementById('go').addEventListener('click', start);
</script>
"""

@app.get("/bridge")
def bridge():
    return HTMLResponse(content=HTML_BRIDGE)

@app.get("/")
def root():
    return PlainTextResponse("OK. See /health, /bridge, /recognize, /pull.")

# ------------------------------- errors ---------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"ok": False, "detail": exc.detail})

@app.exception_handler(Exception)
async def unknown_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"ok": False, "detail": "server_error"})
