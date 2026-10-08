"""Embedding models.

HashingEmbedder is a dependency-free stand-in for a neural embedding model: it hashes word
unigrams/bigrams and character trigrams into a fixed-size vector (the "hashing trick"),
applies sublinear TF weighting and L2-normalises. It captures lexical + sub-word similarity,
not true semantics, but it is fast, deterministic and good enough to learn the mechanics.

OpenAIEmbedder calls a real embedding API (works with OpenAI-compatible servers too).
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import httpx
import numpy as np

from .config import Settings
from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokenize(text, drop_stopwords=True)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
            feats += [f"#{t[i:i + 3]}" for t in toks if len(t) > 4 for i in range(len(t) - 2)]
            counts: dict[int, float] = {}
            for f in feats:
                idx, sign = self._bucket(f)
                weight = 0.5 if f.startswith("#") else 1.0
                counts[idx] = counts.get(idx, 0.0) + sign * weight
            for idx, v in counts.items():
                out[row, idx] = np.sign(v) * np.log1p(abs(v))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", base_url: str | None = None):
        self.model = model
        self.name = f"openai:{model}"
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL",
                                                    "https://api.openai.com/v1")).rstrip("/")
        self.http = httpx.Client(timeout=60, headers={
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
        self.dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 96):  # batch to respect request limits
            r = self.http.post(f"{self.base_url}/embeddings",
                               json={"model": self.model, "input": texts[i:i + 96]})
            r.raise_for_status()
            vecs += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        arr = np.asarray(vecs, dtype=np.float32)
        self.dim = arr.shape[1]
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "openai":
        return OpenAIEmbedder(model=os.environ.get("GROUNDED_EMBED_MODEL",
                                                   "text-embedding-3-small"))
    return HashingEmbedder()
