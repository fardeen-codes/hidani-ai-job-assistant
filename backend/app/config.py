"""
Configuration settings for Hidani AI Job Assistant.
Loads environment variables and resolves absolute paths for data storage.
"""

import os
from pathlib import Path
from typing import List, Optional

# Base Directories
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"

# Attempt to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    for env_path in [BACKEND_DIR / ".env", PROJECT_ROOT / ".env", Path(".env")]:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break
except ImportError:
    pass

# Data file paths
JOBS_FILE = DATA_DIR / "jobs.json"

# In serverless environments like Vercel, the local filesystem is read-only except /tmp
IS_SERVERLESS = os.getenv("VERCEL") == "1" or os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None
if IS_SERVERLESS or not os.access(DATA_DIR if DATA_DIR.exists() else BACKEND_DIR, os.W_OK):
    APPLICATIONS_FILE = Path("/tmp") / "applications.json"
else:
    APPLICATIONS_FILE = DATA_DIR / "applications.json"

# Ensure data directory and files exist
try:
    if not IS_SERVERLESS:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not JOBS_FILE.exists():
        with open(JOBS_FILE, "w", encoding="utf-8") as f:
            f.write("[]")
except OSError:
    pass

try:
    if not APPLICATIONS_FILE.exists():
        with open(APPLICATIONS_FILE, "w", encoding="utf-8") as f:
            f.write("[]")
except OSError:
    pass

# OpenAI / LLM Configuration (Supports Groq, Gemini, OpenRouter, etc.)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL", "").strip() or None

# Server Configuration
APP_TITLE: str = "Hidani AI Job Assistant"
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

# CORS Origins
CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "*"
]

# Validation
def is_openai_configured() -> bool:
    """Check if OpenAI API key is present."""
    return bool(OPENAI_API_KEY and len(OPENAI_API_KEY) > 10)
