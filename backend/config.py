"""
Central configuration for the whole application.

Every tunable value lives here and is read from environment variables
(loaded from a .env file), so nothing important is hardcoded in the
business logic and no secret ever reaches the repository.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains `backend/`, `frontend/`, `.env` ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# load_dotenv does NOT overwrite variables that are already set in the real
# environment, which is exactly what we want for production deployments.
load_dotenv(PROJECT_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to a default."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Settings:
    """Plain settings object. Read once at import time."""

    # ---------- LLM provider ----------
    # "mock" | "openai" | "anthropic"
    # "mock" needs no API key, so the full RAG pipeline can be tested offline.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "").strip()
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "").strip()
    # Optional: point the OpenAI client at any OpenAI-compatible server
    # (Groq, Together, OpenRouter, a local Ollama, ...).
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "").strip()
    LLM_TEMPERATURE: float = _get_float("LLM_TEMPERATURE", 0.0)
    LLM_MAX_TOKENS: int = _get_int("LLM_MAX_TOKENS", 600)

    # ---------- Embeddings ----------
    # all-MiniLM-L6-v2: 384 dimensions, ~80 MB, fast on CPU. Good default.
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

    # ---------- RAG parameters ----------
    # Chunk size is measured in CHARACTERS (simple and predictable).
    CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 1000)
    CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 200)
    TOP_K: int = _get_int("TOP_K", 4)
    # Chunks scoring below this cosine similarity are treated as irrelevant.
    # 0.0 disables the filter. ~0.25 is a reasonable starting point.
    MIN_SIMILARITY: float = _get_float("MIN_SIMILARITY", 0.20)

    # ---------- Uploads ----------
    MAX_UPLOAD_MB: int = _get_int("MAX_UPLOAD_MB", 20)
    DATA_DIR: Path = PROJECT_ROOT / "data"

    # ---------- API ----------
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1").strip()
    API_PORT: int = _get_int("API_PORT", 8000)


settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
