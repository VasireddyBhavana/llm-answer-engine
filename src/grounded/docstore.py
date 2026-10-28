"""Document store: ingest files, persist chunks + embeddings in SQLite, serve a live Retriever.

Ingestion pipeline (one transaction per document):
    bytes -> size/type checks -> sha256 dedupe -> parse -> chunk -> embed -> store
The in-memory Retriever is rebuilt lazily after any change. That is fine for thousands of
chunks; at millions you would keep an ANN index (HNSW / pgvector) updated incrementally.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .auth import EVERYONE, Principal
from .db import Database
from .embeddings import Embedder
from .ingest import Chunk, chunk_document
from .loaders import SUPPORTED, parse_file
from .retrieval import Retriever


@dataclass
class Document:
    id: str
    title: str
    filename: str
    mime: str
    sha256: str
    size_bytes: int
    groups: list[str]
    tags: list[str]
    owner_id: str | None
    n_chunks: int
    created_at: float
    meta: dict = field(default_factory=dict)
    duplicate: bool = False  # True when an identical upload already existed

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentTooLarge(ValueError):
    pass


def _clean_labels(values: list[str] | None) -> list[str]:
    return sorted({v.strip().lower() for v in (values or []) if v and v.strip()})


class DocumentStore:
    def __init__(self, db: Database, embedder: Embedder, chunk_size: int = 120,
                 chunk_overlap: int = 20, max_upload_mb: int = 20):
        self.db, self.embedder = db, embedder
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.max_bytes = max_upload_mb * 1024 * 1024
        self._lock = threading.RLock()
        self._retriever: Retriever | None = None
        self._ensure_embedder_matches()

    # -- writes ----------------------------------------------------------------
    def add(self, filename: str, data: bytes, groups: list[str] | None = None,
            tags: list[str] | None = None, owner: Principal | None = None,
            doc_id: str | None = None) -> Document:
        if len(data) > self.max_bytes:
            raise DocumentTooLarge(f"file is {len(data) / 1e6:.1f} MB; limit is "
                                   f"{self.max_bytes / 1e6:.0f} MB")
        groups = _clean_labels(groups) or [EVERYONE]
        tags = _clean_labels(tags)
        sha = hashlib.sha256(data).hexdigest()

        # Same bytes shared with the same groups -> idempotent re-upload, no new rows.
        for row in self.db.query("SELECT * FROM documents WHERE sha256 = ?", (sha,)):
            if json.loads(row["groups_json"]) == groups:
                doc = self._row_to_doc(row)
                doc.duplicate = True
                return doc

        parsed = parse_file(filename, data)
        doc_id = doc_id or "d_" + uuid.uuid4().hex[:12]
        chunks = chunk_document(doc_id, parsed.text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError("document produced no chunks")
        vectors = self.embedder.embed([c.text for c in chunks]).astype(np.float32)

        doc = Document(doc_id, parsed.title, Path(filename).name, parsed.mime, sha, len(data),
                       groups, tags, owner.user_id if owner else None, len(chunks), time.time(),
                       parsed.meta)
        with self._lock, self.db.tx() as c:
            c.execute("INSERT INTO documents(id, title, filename, mime, sha256, size_bytes, "
                      "groups_json, tags_json, meta_json, owner_id, n_chunks, created_at) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (doc.id, doc.title, doc.filename, doc.mime, sha, doc.size_bytes,
                       json.dumps(groups), json.dumps(tags), json.dumps(doc.meta),
                       doc.owner_id, doc.n_chunks, doc.created_at))
            c.executemany("INSERT INTO chunks(id, doc_id, ord, section, text, vector) "
                          "VALUES (?, ?, ?, ?, ?, ?)",
                          [(ch.id, doc_id, i, ch.section, ch.text, vectors[i].tobytes())
                           for i, ch in enumerate(chunks)])
            self._retriever = None
        return doc

    def add_path(self, path: Path, **kw) -> Document:
        return self.add(path.name, path.read_bytes(), **kw)

    def add_folder(self, folder: Path, **kw) -> list[Document]:
        return [self.add_path(p, **kw) for p in sorted(folder.glob("**/*"))
                if p.is_file() and p.suffix.lower() in SUPPORTED]

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            n = self.db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._retriever = None
        return n > 0

    # -- reads -----------------------------------------------------------------
    def get(self, doc_id: str) -> Document | None:
        row = self.db.one("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return self._row_to_doc(row) if row else None

    def list(self, principal: Principal | None = None, tag: str | None = None) -> list[Document]:
        docs = [self._row_to_doc(r) for r in
                self.db.query("SELECT * FROM documents ORDER BY created_at DESC")]
        if principal is not None:
            docs = [d for d in docs if principal.can_read(d.groups)]
        if tag:
            docs = [d for d in docs if tag.lower() in d.tags]
        return docs

    def count(self) -> int:
        return self.db.one("SELECT COUNT(*) AS n FROM documents")["n"]

    def retriever(self) -> Retriever:
        with self._lock:
            if self._retriever is None:
                self._retriever = self._build_retriever()
            return self._retriever

    # -- internals ---------------------------------------------------------------
    def _build_retriever(self) -> Retriever:
        rows = self.db.query(
            "SELECT c.id, c.doc_id, c.section, c.text, c.vector, d.title, d.groups_json, "
            "d.tags_json FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "ORDER BY d.created_at, c.ord")
        chunks = [Chunk(r["id"], r["doc_id"], r["title"], r["section"], r["text"],
                        json.loads(r["groups_json"]), json.loads(r["tags_json"])) for r in rows]
        vectors = (np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
                   if rows else None)
        return Retriever(chunks, self.embedder, vectors)

    def _ensure_embedder_matches(self) -> None:
        """Vectors from different embedding models are not comparable: re-embed on change."""
        stored = self.db.get_meta("embedder")
        if stored == self.embedder.name:
            return
        rows = self.db.query("SELECT id, text FROM chunks")
        if rows:
            vecs = self.embedder.embed([r["text"] for r in rows]).astype(np.float32)
            with self.db.tx() as c:
                c.executemany("UPDATE chunks SET vector = ? WHERE id = ?",
                              [(vecs[i].tobytes(), r["id"]) for i, r in enumerate(rows)])
        self.db.set_meta("embedder", self.embedder.name)

    @staticmethod
    def _row_to_doc(row) -> Document:
        return Document(row["id"], row["title"], row["filename"], row["mime"], row["sha256"],
                        row["size_bytes"], json.loads(row["groups_json"]),
                        json.loads(row["tags_json"]), row["owner_id"], row["n_chunks"],
                        row["created_at"], json.loads(row["meta_json"]))
