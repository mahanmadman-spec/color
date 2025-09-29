import os, io, time, threading, queue
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from vosk import Model, KaldiRecognizer
import soundfile as sf

APP_PORT = int(os.getenv("PORT", "8000"))
VOCAB     = set([s.strip() for s in open("vocab/colors_de.txt", "r", encoding="utf-8").read().splitlines() if s.strip()])
MODEL_DIR = os.getenv("VOSK_MODEL_DIR", "models/vosk-model-small-de-0.15")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# token queues keyed by short code
QUEUES: dict[str, "queue.Queue[str]"] = {}
def q_for(code: str) -> "queue.Queue[str]":
    if code not in QUEUES: QUEUES[code] = queue.Queue()
    return QUEUES[code]

MODEL = Model(MODEL_DIR)

@app.get("/health")
def health():
    return {"ok": True, "model": os.path.basename(MODEL_DIR), "vocab": len(VOCAB)}

@app.post("/recognize")
async def recognize(code: str = Form(...), audio: UploadFile = File(...)):
    """
    Accepts a mono 16-kHz WAV/OGG/WEBM chunk. Emits the best single-token match in our vocabulary.
    Pushes the token into the client's queue (code).
    """
    raw = await audio.read()
    # decode to mono 16k float32 using soundfile
    data, sr = sf.read(io.BytesIO(raw), dtype="int16", always_2d=False)
    if sr != 16000:
        # resample quickly via soundfile if needed (simple fallback)
        data = sf.resample(data, sr, 16000)
        sr = 16000

    rec = KaldiRecognizer(MODEL, 16000)
    ok = rec.AcceptWaveform(data.tobytes())
    txt = rec.FinalResult() if ok else rec.PartialResult()
    # pick first vocab word present (very constrained)
    import json, re
    token = None
    try:
        j = json.loads(txt)
        hyp = (j.get("text") or j.get("partial") or "").strip().lower()
        hyp = re.sub(r"[^a-zäöüß\s-]", " ", hyp)
        # best-effort alias normalization; keep it short
        aliases = {"weiss":"weiß","gruen":"grün","tuerkis":"türkis","hell-blau":"hellblau","dunkel-blau":"dunkelblau","hell-gruen":"hellgrün","dunkel-gruen":"dunkelgrün","purple":"violett","violet":"violett","turkis":"türkis"}
        hyp = aliases.get(hyp, hyp)
        # choose a single word token that exists in VOCAB
        for w in hyp.split():
            if w in VOCAB:
                token = w
                break
    except Exception:
        pass

    if token:
        q_for(code).put(token)
        return {"ok": True, "token": token}
    return {"ok": True, "token": None}

@app.get("/pull")
def pull(code: str):
    """Polled by Roblox client. Returns and clears queued tokens."""
    q = q_for(code)
    items = []
    try:
        while True:
            items.append(q.get_nowait())
    except Exception:
        pass
    return {"tokens": items}

@app.get("/bridge")
def bridge():
    # very small recorder UI; records 1s webm chunks and POSTs to /recognize with the same code
    html = """
<!doctype html><meta charset="utf-8">
<title>Mic-Bridge</title>
<style> body{font:16px system-ui,Segoe UI,Arial;margin:24px;max-width:720px} #log{white-space:pre-wrap;background:#111;color:#eee;padding:12px;border-radius:8px} button{padding:10px 14px;font-weight:600}</style>
<h2>Mic-Bridge</h2>
<p>Enter the 8-char code shown in Roblox, allow microphone, leave this tab open.</p>
<p><input id="code" placeholder="XXXXXXXX" maxlength="16" style="font:700 20px monospace;width:12ch;text-transform:uppercase"> <button id="go">Start</button> <span id="hint"></span></p>
<div id="log"></div>
<script>
const base = location.origin;
const $ = sel => document.querySelector(sel);
const log = (m)=>{ const d=$("#log"); d.textContent = m+"\\n"+d.textContent; }

let media, rec, code="", running=false;

async function start(){
  code = ($("#code").value||"").trim();
  if(!code){ alert("Enter code"); return; }
  $("#hint").textContent = "Recording… keep this tab open.";
  media = await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000,channelCount:1}, video:false});
  rec = new MediaRecorder(media, {mimeType:"audio/webm"});
  rec.ondataavailable = async ev=>{
    if(!ev.data || ev.data.size < 2000) return;
    const fd = new FormData();
    fd.append("code", code);
    fd.append("audio", ev.data, "chunk.webm");
    try {
      const r = await fetch(base+"/recognize", {method:"POST", body:fd});
      const j = await r.json();
      if(j.token){ log("→ "+j.token); }
    } catch(e){}
  };
  rec.start(1000); // 1s chunks
  running = true;
}

$("#go").onclick = start;
</script>
"""
    return HTMLResponse(content=html)
