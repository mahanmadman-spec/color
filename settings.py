import os

# change: pick a model (small = faster)
MODEL_URL = os.getenv("MODEL_URL", "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip")  # change
MODEL_DIR = os.getenv("MODEL_DIR", "./models")  # change
MODEL_SUBDIR = os.getenv("MODEL_SUBDIR", "vosk-model-small-de-0.15")  # change
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))  # change
VOCAB_PATH = os.getenv("VOCAB_PATH", "./vocab/de_colors.txt")  # change
CHUNK_MS = int(os.getenv("CHUNK_MS", "200"))  # change
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "64"))  # change
