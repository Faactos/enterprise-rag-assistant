"""Runtime configuration.

Every retrieval knob is env-overridable so `eval/sweep.py` can vary a setting
without editing code -- the sweep runs this module's defaults through
`dataclasses.replace`, which is only safe because Settings is frozen.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load .env before any os.getenv default below is evaluated. Without this the
# keys in .env are invisible to the process and GROQ_API_KEY appears unset.
# The path is explicit: find_dotenv() walks the stack and breaks when a module
# is imported from stdin or a stdio child process.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ModuleNotFoundError:  # dotenv is optional; real env vars still work
    pass


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # storage
    data_dir: str = os.getenv("DATA_DIR", str(ROOT / "data"))
    db_path: str = os.getenv("DB_PATH", str(ROOT / "data" / "meta.db"))
    upload_dir: str = os.getenv("UPLOAD_DIR", str(ROOT / "data" / "uploads"))
    qdrant_path: str = os.getenv("QDRANT_PATH", str(ROOT / "data" / "qdrant"))
    qdrant_url: str | None = os.getenv("QDRANT_URL") or None
    collection: str = os.getenv("QDRANT_COLLECTION", "chunks")

    # models
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY") or None
    max_tokens: int = int(os.getenv("MAX_TOKENS", "900"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))

    # retrieval -- what the sweep varies
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    candidate_k: int = int(os.getenv("CANDIDATE_K", "20"))
    use_hybrid: bool = _flag("USE_HYBRID", True)
    use_reranker: bool = _flag("USE_RERANKER", True)
    min_score: float = float(os.getenv("MIN_SCORE", "0.25"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))


settings = Settings()
