"""Central configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # Which LLM backend to use: "mock" (offline, deterministic), "anthropic", or "openai"
    # ("openai" also works with any OpenAI-compatible server: Ollama, vLLM, Groq, LM Studio...).
    llm_provider: str = field(default_factory=lambda: _env("GROUNDED_LLM", "mock"))
    model: str = field(default_factory=lambda: _env("GROUNDED_MODEL", ""))
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    # Embeddings: "hashing" (offline, no downloads) or "openai".
    embedder: str = field(default_factory=lambda: _env("GROUNDED_EMBEDDER", "hashing"))

    # Retrieval knobs — these are the things you "hill climb" with `grounded eval`.
    # chunk size and overlap are measured in words
    chunk_size: int = field(default_factory=lambda: int(_env("GROUNDED_CHUNK_SIZE", "120")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("GROUNDED_CHUNK_OVERLAP", "20")))
    top_k: int = field(default_factory=lambda: int(_env("GROUNDED_TOP_K", "4")))
    retriever: str = field(default_factory=lambda: _env("GROUNDED_RETRIEVER", "hybrid"))

    data_dir: Path = field(default_factory=lambda: Path(_env("GROUNDED_DATA_DIR", "data")))
    index_dir: Path = field(default_factory=lambda: Path(_env("GROUNDED_INDEX_DIR", ".index")))
    trace_file: Path = field(
        default_factory=lambda: Path(_env("GROUNDED_TRACE_FILE", ".traces/traces.jsonl"))
    )
    memory_file: Path = field(
        default_factory=lambda: Path(_env("GROUNDED_MEMORY_FILE", ".memory/users.json"))
    )

    max_agent_steps: int = field(default_factory=lambda: int(_env("GROUNDED_MAX_AGENT_STEPS", "6")))


def get_settings() -> Settings:
    return Settings()
