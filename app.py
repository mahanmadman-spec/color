# app.py

import os, io, time, json, queue, zipfile, tempfile, shutil, urllib.request
from typing import Dict, Set, Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from vosk import Model, KaldiRecognizer
import soundfile as sf

START_TS = time.monotonic()

# VALUES TO CHANGE ONLY (optional: override via env)
MODEL_DIR  = os.getenv("VOSK_MODEL_DIR", "models/vosk-model-small-de-0.15")
MODEL_URL  = os.getenv("VOSK_MODEL_URL", "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip")
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab/colors_de.txt")
# ================================================

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

def load_vocab(path: str) -> Set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            items = {s.strip().lower() for s in f.read().splitlines() if s.strip()}
            return items if items else set(DEFAULT_VOCAB)
    except FileNotFoundError:
        return set(DEFAULT_VOCAB)

VOCAB: Set[str] = load_vocab(VOCAB_PATH)

def normalize_token(s: str) -> Optional[str]:
    if not s: return None
    s = s.strip().lower()
    s = "".join(ch if ("a" <= ch <= "z" or ch in " äöüß-") else " " for ch in s)
    s = " ".join(s.split())
    if not s: return None
    s = ALIASES.get(s, s)
    parts = s.replace("-", "").split()
    for w in parts:
        w = ALIASES.get(w, w)
        if w in VOCAB:
            return w
    return None

def model_ok(path: str) -> bool:
    return os.path.exists(os.path.join(path, "graph", "Gr.fst"))

def ensure_model(path: str, url: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if model_ok(path): return
    tmp_zip = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp_zip = tmp.name
            with urllib.request.urlopen(url, timeout=300) as r:
                shutil.copyfileobj(r, tmp)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            extract_root = os.path.abspath(os.path.join(path, os.pardir))
            zf.extractall(extract_root)
        if not model_ok(path):
            parent = os.path.abspath(os.path.join(path, os.pardir))
            for name in os.listdir(parent):
                cand = os.path.join(parent, name)
                if os.path.isdir(cand) and model_ok(cand):
                    if os.path.abspath(cand) != os.path.abspath(path):
                        try: os.replace(cand, path)
                        except Exception: pass
                    break
        if not model_ok(path):
            raise RuntimeError("model extracted but invalid structure")
    finally:
        if tmp_zip and os.path.exists(tmp_zip):
            try: os.remove(tmp_zip)
            except Exception: pass

ensure_model(MODEL_DIR, MODEL_URL)
MODEL = Model(MODEL_DIR)

QUEUES: Dict[str, "queue.Queue[str]"] = {}
QLEN: Dict[str, int] = {}

def q_for(key: str) -> "queue.Queue[str]":
    key = key.strip()
    if key not in QUEUES:
        QUEUES[key] = queue.Queue()
        QLEN[key] = 0
    return QUEUES[key]

def push_token(key: str, token: str) -> None:
    q = q_for(key)
    if QLEN.get(key, 0) >= MAX_QUEUE_LEN_PER_CODE:
        try:
            q.get_nowait()
            QLEN[key] = max(0, QLEN.get(key, 0) - 1)
        except Exception: pass
    q.put(token)
    QLEN[key] = QLEN.get(key, 0) + 1

def drain_tokens(key: str) -> List[str]:
    out: List[str] = []
    q = q_for(key)
    try:
        while True:
            out.append(q.get_nowait())
            QLEN[key] = max(0, QLEN.get(key, 0) - 1)
    except Exception: pass
    return out

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

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
async def recognize(
    code: Optional[str] = Form(None),
    uid:  Optional[str] = Form(None),
    audio: UploadFile = File(...),
):
    key = (code or uid or "").strip()
    if not key or len(key) > 64:
        raise HTTPException(status_code=400, detail="invalid code")
    raw = await audio.read()
    if not raw or len(raw) < 512:
        return {"ok": True, "token": None}
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
        push_token(key, token)
        return {"ok": True, "token": token}
    return {"ok": True, "token": None}

@app.get("/pull")
def pull(code: Optional[str] = Query(None), uid: Optional[str] = Query(None)):
    key = (code or uid or "").strip()
    if not key or len(key) > 64:
        raise HTTPException(status_code=400, detail="invalid code")
    return {"tokens": drain_tokens(key)}

HTML_BRIDGE = """
<!doctype html><meta charset="utf-8">
<title>Mic-Bridge</title>
<style>
body{font:16px system-ui,Segoe UI,Arial;margin:24px;max-width:880px;background:#0b0b0f;color:#eaeaf0}
h1{font-size:22px;margin:0 0 14px}
label,input,button{font:inherit}
input#code{font:700 22px ui-monospace,Consolas,monospace;width:14ch;text-transform:none;padding:6px 8px;border-radius:8px;border:1px solid #334}
button{padding:8px 14px;border-radius:10px;border:1px solid #46f;background:#9df;color:#012;font-weight:700;cursor:pointer}
#hint{margin-left:12px;opacity:.85}
#log{white-space:pre-wrap;background:#10131a;border:1px solid #233;padding:12px;border-radius:10px;margin-top:16px;min-height:120px}
.small{opacity:.8;font-size:13px}
kbd{background:#222;border:1px solid #444;padding:2px 5px;border-radius:4px}
</style>
<h1>Mic-Bridge</h1>
<p>Enter your in-game code (e.g. your Roblox UserId). Keep this tab open.</p>
<p>
  <label for="code">Code:</label>
  <input id="code" placeholder="1192628416" maxlength="32">
  <button id="go">Start</button>
  <span id="hint"></span>
</p>
<div class="small">Allow microphone access when prompted.</div>
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

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"ok": False, "detail": exc.detail})

@app.exception_handler(Exception)
async def unknown_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"ok": False, "detail": "server_error"})
