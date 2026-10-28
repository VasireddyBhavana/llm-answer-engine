"""Central configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_list(name: str, default: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in _env(name, default).split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    # ---------------------------------------------------------------- models
    # "mock" (offline, deterministic), "anthropic", or "openai" (also any OpenAI-compatible
    # server: Ollama, vLLM, Groq, LM Studio...).
    llm_provider: str = field(default_factory=lambda: _env("GROUNDED_LLM", "mock"))
    model: str = field(default_factory=lambda: _env("GROUNDED_MODEL", ""))
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    # "hashing" (offline, no downloads) or "openai".
    embedder: str = field(default_factory=lambda: _env("GROUNDED_EMBEDDER", "hashing"))

    # ---------------------------------------------------------------- retrieval
    # Knobs to hill-climb with `grounded eval`. Chunk size and overlap are in words.
    chunk_size: int = field(default_factory=lambda: _env_int("GROUNDED_CHUNK_SIZE", 120))
    chunk_overlap: int = field(default_factory=lambda: _env_int("GROUNDED_CHUNK_OVERLAP", 20))
    top_k: int = field(default_factory=lambda: _env_int("GROUNDED_TOP_K", 4))
    # bm25 | vector | hybrid | hybrid+rerank
    retriever: str = field(default_factory=lambda: _env("GROUNDED_RETRIEVER", "hybrid"))

    # ---------------------------------------------------------------- storage
    data_dir: Path = field(default_factory=lambda: Path(_env("GROUNDED_DATA_DIR", "data")))
    db_path: Path = field(
        default_factory=lambda: Path(_env("GROUNDED_DB", ".grounded/grounded.db")))
    trace_file: Path = field(
        default_factory=lambda: Path(_env("GROUNDED_TRACE_FILE", ".grounded/traces.jsonl")))
    memory_file: Path = field(
        default_factory=lambda: Path(_env("GROUNDED_MEMORY_FILE", ".grounded/memory.json")))
    # Seed data/corpus into the store (readable by everyone) when the store is empty.
    seed_corpus: bool = field(default_factory=lambda: _env("GROUNDED_SEED_CORPUS", "1") == "1")
    max_upload_mb: int = field(default_factory=lambda: _env_int("GROUNDED_MAX_UPLOAD_MB", 20))

    # ---------------------------------------------------------------- auth
    # Set a long random value in production. If empty, a random per-process secret is used,
    # which means tokens stop working when the server restarts.
    jwt_secret: str = field(default_factory=lambda: _env("GROUNDED_JWT_SECRET", ""))
    jwt_ttl_minutes: int = field(default_factory=lambda: _env_int("GROUNDED_JWT_TTL_MIN", 60))
    admin_email: str = field(default_factory=lambda: _env("GROUNDED_ADMIN_EMAIL", ""))
    admin_password: str = field(default_factory=lambda: _env("GROUNDED_ADMIN_PASSWORD", ""))
    rate_limit_per_min: int = field(
        default_factory=lambda: _env_int("GROUNDED_RATE_LIMIT_PER_MIN", 60))

    # ---------------------------------------------------------------- agent + integrations
    max_agent_steps: int = field(default_factory=lambda: _env_int("GROUNDED_MAX_AGENT_STEPS", 6))
    integrations: tuple[str, ...] = field(
        default_factory=lambda: _env_list("GROUNDED_INTEGRATIONS", "weather,wikipedia,webhook"))
    webhook_url: str = field(default_factory=lambda: _env("GROUNDED_WEBHOOK_URL", ""))
    http_timeout_s: float = field(
        default_factory=lambda: float(_env("GROUNDED_HTTP_TIMEOUT_S", "8")))


def get_settings() -> Settings:
    return Settings()
