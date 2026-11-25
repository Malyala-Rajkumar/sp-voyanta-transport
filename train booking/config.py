import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

# Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

# RailKit Configuration
RAILKIT_API_KEY = os.getenv("RAILKIT_API_KEY", "").strip()

RAILKIT_BASE_URL = os.getenv(
    "RAILKIT_BASE_URL",
    "https://api.railkit.in"
)

# Validate configuration
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY / GEMINI_API_KEY is missing from environment")

if not RAILKIT_API_KEY:
    raise ValueError("RAILKIT_API_KEY is missing from environment")