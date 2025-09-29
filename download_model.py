# Ensures the model is present at $VOSK_MODEL_DIR by downloading the zip from $VOSK_MODEL_URL
import os, zipfile, tempfile, shutil, urllib.request, sys

MODEL_DIR = os.getenv("VOSK_MODEL_DIR", "models/vosk-model-small-de-0.15")
MODEL_URL = os.getenv("VOSK_MODEL_URL", "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip")

def model_ok(path: str) -> bool:
    return os.path.exists(os.path.join(path, "graph", "Gr.fst"))

def ensure_model(path: str, url: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if model_ok(path):
        print(f"[download_model] model already present: {path}")
        return
    tmp_zip = None
    try:
        print(f"[download_model] downloading {url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp_zip = tmp.name
            with urllib.request.urlopen(url, timeout=300) as r:
                shutil.copyfileobj(r, tmp)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            extract_root = os.path.abspath(os.path.join(path, os.pardir))
            zf.extractall(extract_root)
        # try rename if needed
        if not model_ok(path):
            parent = os.path.abspath(os.path.join(path, os.pardir))
            for name in os.listdir(parent):
                cand = os.path.join(parent, name)
                if os.path.isdir(cand) and model_ok(cand):
                    if os.path.abspath(cand) != os.path.abspath(path):
                        os.replace(cand, path)
                    break
        if not model_ok(path):
            raise SystemExit("[download_model] extracted, but model structure invalid")
        print(f"[download_model] ready at {path}")
    finally:
        if tmp_zip and os.path.exists(tmp_zip):
            try: os.remove(tmp_zip)
            except Exception: pass

if __name__ == "__main__":
    ensure_model(MODEL_DIR, MODEL_URL)
