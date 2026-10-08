from pathlib import Path

import pytest

from grounded.config import Settings
from grounded.llm import MockLLM
from grounded.rag import AnswerEngine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="mock",
        embedder="hashing",
        data_dir=ROOT / "data",
        index_dir=tmp_path / "index",
        trace_file=tmp_path / "traces.jsonl",
        memory_file=tmp_path / "memory.json",
    )


@pytest.fixture
def engine(settings: Settings) -> AnswerEngine:
    return AnswerEngine(settings, llm=MockLLM())
