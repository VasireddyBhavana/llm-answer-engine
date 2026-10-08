"""grounded — a small, readable answer engine: hybrid RAG with citations, a tool-using agent,
personalization memory, tracing, and an evaluation harness."""

from .agent import Agent
from .rag import AnswerEngine

__all__ = ["Agent", "AnswerEngine"]
__version__ = "0.1.0"
