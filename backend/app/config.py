import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve .env relative to this file: backend/app/config.py → ../../.. → anomaly-explainer-agent/
# This works regardless of the working directory uvicorn is started from.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

PLOT_DIR = Path(os.getenv("PLOT_DIR", "/tmp/anomaly-plots"))
SESSION_TTL = int(os.getenv("SESSION_TTL", "3600"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MAX_MODEL_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

PLOT_DIR.mkdir(parents=True, exist_ok=True)
