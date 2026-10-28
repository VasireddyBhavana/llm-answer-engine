"""Retrieval: BM25 (sparse), vector (dense), hybrid via Reciprocal Rank Fusion, and MMR.

Everything is in-process numpy so you can read every line. In production you would swap the
VectorIndex for FAISS / pgvector / Qdrant and BM25 for Elasticsearch / OpenSearch / Tantivy —
the interfaces stay the same.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder
from .ingest import Chunk
from .text import tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float
    retriever: str


# --------------------------------------------------------------------------- BM25


class BM25Index:
    """Okapi BM25. score(q, d) = sum_t IDF(t) * tf*(k1+1) / (tf + k1*(1 - b + b*|d|/avgdl))."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c.text, drop_stopwords=True)) for c in chunks]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avgdl = float(self.lens.mean()) if len(chunks) else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5, mask: np.ndarray | None = None) -> list[Hit]:
        q = tokenize(query, drop_stopwords=True)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for t in set(q):
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.docs], dtype=np.float32)
            denom = tf + self.k1 * (1 - self.b + self.b * self.lens / max(self.avgdl, 1e-9))
            scores += self.idf[t] * tf * (self.k1 + 1) / np.where(denom == 0, 1, denom)
        if mask is not None:
            scores[~mask] = 0.0  # pre-filter: forbidden chunks can never be ranked
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), "bm25") for i in order if scores[i] > 0]


# --------------------------------------------------------------------------- Dense


class VectorIndex:
    """Exact cosine search over L2-normalised vectors (a matrix multiply)."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder,
                 vectors: np.ndarray | None = None):
        self.chunks, self.embedder = chunks, embedder
        if vectors is None:
            vectors = (embedder.embed([c.text for c in chunks]) if chunks
                       else np.zeros((0, getattr(embedder, "dim", 1) or 1), dtype=np.float32))
        self.vectors = vectors

    def search(self, query: str, k: int = 5, mask: np.ndarray | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        sims = self.vectors @ q
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], float(sims[i]), "vector") for i in order
                if np.isfinite(sims[i])]


# --------------------------------------------------------------------------- Fusion + MMR


def reciprocal_rank_fusion(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """RRF: score(d) = sum over lists of 1 / (k + rank(d)). Uses ranks only, so no calibration."""
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits, start=1):
            fused[h.chunk.id] = fused.get(h.chunk.id, 0.0) + 1.0 / (k + rank)
            by_id[h.chunk.id] = h.chunk
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    return [Hit(by_id[cid], s, "hybrid") for cid, s in ranked]


def mmr(query_vec: np.ndarray, hits: list[Hit], vectors: dict[str, np.ndarray], k: int,
        lam: float = 0.7) -> list[Hit]:
    """Maximal Marginal Relevance: pick relevant hits that are not redundant with earlier picks."""
    selected: list[Hit] = []
    pool = list(hits)
    while pool and len(selected) < k:
        def gain(h: Hit) -> float:
            rel = float(vectors[h.chunk.id] @ query_vec)
            red = max((float(vectors[h.chunk.id] @ vectors[s.chunk.id]) for s in selected),
                      default=0.0)
            return lam * rel - (1 - lam) * red
        best = max(pool, key=gain)
        selected.append(best)
        pool.remove(best)
    return selected


# --------------------------------------------------------------------------- Facade


class Retriever:
    """Single entry point used by the RAG pipeline, the agent and the evals."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder,
                 vectors: np.ndarray | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25Index(chunks)
        self.vector = VectorIndex(chunks, embedder, vectors)
        self._vec_by_id = {c.id: self.vector.vectors[i] for i, c in enumerate(chunks)}

    def build_mask(self, allow: Callable[[Chunk], bool] | None = None,
                   tags: list[str] | None = None,
                   doc_ids: list[str] | None = None) -> np.ndarray | None:
        """Boolean mask of chunks the caller may see AND that match the metadata filters.

        Filtering happens *before* ranking (pre-filtering). Post-filtering a global top-k would
        silently return fewer results for users with narrow access, and is one bug away from
        leaking restricted text into the prompt.
        """
        if allow is None and not tags and not doc_ids:
            return None
        tagset, idset = set(tags or []), set(doc_ids or [])
        return np.array([(allow is None or allow(c))
                         and (not tagset or bool(tagset & set(c.tags)))
                         and (not idset or c.doc_id in idset)
                         for c in self.chunks], dtype=bool)

    def search(self, query: str, k: int = 4, mode: str = "hybrid", use_mmr: bool = False,
               boost_docs: dict[str, float] | None = None, candidates: int = 20,
               allow: Callable[[Chunk], bool] | None = None, tags: list[str] | None = None,
               doc_ids: list[str] | None = None) -> list[Hit]:
        """mode: 'bm25' | 'vector' | 'hybrid'. boost_docs: per-doc multipliers (personalization).
        allow: access-control predicate per chunk. tags / doc_ids: metadata filters."""
        mask = self.build_mask(allow, tags, doc_ids)
        if mode == "bm25":
            hits = self.bm25.search(query, candidates, mask)
        elif mode == "vector":
            hits = self.vector.search(query, candidates, mask)
        elif mode == "hybrid":
            hits = reciprocal_rank_fusion([self.bm25.search(query, candidates, mask),
                                           self.vector.search(query, candidates, mask)])
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        if boost_docs:
            for h in hits:
                h.score *= boost_docs.get(h.chunk.doc_id, 1.0)
            hits.sort(key=lambda h: -h.score)

        if use_mmr:
            qv = self.embedder.embed([query])[0]
            return mmr(qv, hits, self._vec_by_id, k)
        return hits[:k]

    # -- persistence --------------------------------------------------------
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vector.vectors)
        (folder / "chunks.json").write_text(json.dumps(
            {"embedder": self.embedder.name, "chunks": [c.to_dict() for c in self.chunks]},
            indent=1))

    @classmethod
    def load(cls, folder: Path, embedder: Embedder) -> Retriever:
        meta = json.loads((folder / "chunks.json").read_text())
        if meta["embedder"] != embedder.name:
            raise ValueError(f"index built with {meta['embedder']}, but embedder is "
                             f"{embedder.name}; re-run `grounded ingest`")
        chunks = [Chunk(**c) for c in meta["chunks"]]
        return cls(chunks, embedder, np.load(folder / "vectors.npy"))
