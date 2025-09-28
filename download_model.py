import os, sys, zipfile, shutil, tempfile
from pathlib import Path
import requests
from settings import MODEL_URL, MODEL_DIR, MODEL_SUBDIR

def ensure_model():
    model_dir = Path(MODEL_DIR)
    target = model_dir / MODEL_SUBDIR
    if target.exists() and (target / "am" if (target / "am").exists() else target).exists():
        return str(target)
    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "model.zip"
        with requests.get(MODEL_URL, stream=True, timeout=120) as r:  # change: adjust timeout
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk: f.write(chunk)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(td)
        # pick first extracted dir if name unknown
        subdirs = [p for p in Path(td).iterdir() if p.is_dir()]
        src = None
        for p in subdirs:
            if "vosk-model" in p.name:
                src = p; break
        if src is None:
            raise RuntimeError("Model zip did not contain a vosk-model directory")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(src), str(target))
    return str(target)

if __name__ == "__main__":
    print(ensure_model())
