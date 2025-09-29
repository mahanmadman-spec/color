# Color Bridge (German, Vosk)

Endpoints
- GET  /health
- POST /recognize  (form: code, audio=WAV blob)
- GET  /pull?code=XXXX
- GET  /bridge

Deploy on Render
1) Connect repo.
2) Use render.yaml.
3) Ensure env matches defaults (VOSK_MODEL_DIR, VOSK_MODEL_URL).
4) First build downloads and caches the model into models/.
