"""Ingestion: load documents and split them into overlapping, structure-aware chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str          # e.g. "rag#3"
    doc_id: str      # e.g. "rag"
    title: str       # document title
    section: str     # nearest heading
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_documents(folder: Path) -> dict[str, str]:
    docs = {}
    for path in sorted(folder.glob("**/*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def _sections(markdown: str) -> list[tuple[str, str, str]]:
    """Split markdown into (title, section_heading, body) using headings as boundaries."""
    title, heading, buf, out = "", "", [], []
    for line in markdown.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if buf:
                out.append((title, heading, "\n".join(buf).strip()))
                buf = []
            if len(m.group(1)) == 1 and not title:
                title = m.group(2).strip()
            heading = m.group(2).strip()
        else:
            buf.append(line)
    if buf:
        out.append((title, heading, "\n".join(buf).strip()))
    return [s for s in out if s[2]]


def chunk_document(doc_id: str, text: str, chunk_size: int = 120,
                   overlap: int = 20) -> list[Chunk]:
    """Structure-aware chunking.

    1. Split on headings so a chunk never spans two sections.
    2. Pack whole paragraphs into chunks of ~chunk_size words.
    3. If a single paragraph is too long, fall back to a sliding word window with `overlap`.
    Each chunk is prefixed with its title/section so it embeds with context
    (a cheap version of "contextual chunk headers").
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []

    def emit(title: str, section: str, body: str) -> None:
        header = f"{title} > {section}" if section and section != title else title
        chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, title or doc_id, section,
                            f"[{header}]\n{body}".strip()))

    for title, section, body in _sections(text):
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buf: list[str] = []
        for p in paras:
            words = p.split()
            if len(words) > chunk_size:  # long paragraph -> sliding window
                if buf:
                    emit(title, section, "\n\n".join(buf))
                    buf = []
                step = chunk_size - overlap
                for i in range(0, len(words), step):
                    emit(title, section, " ".join(words[i:i + chunk_size]))
                    if i + chunk_size >= len(words):
                        break
                continue
            if sum(len(b.split()) for b in buf) + len(words) > chunk_size and buf:
                emit(title, section, "\n\n".join(buf))
                buf = []
            buf.append(p)
        if buf:
            emit(title, section, "\n\n".join(buf))
    return chunks


def chunk_corpus(docs: dict[str, str], chunk_size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for doc_id, text in docs.items():
        out.extend(chunk_document(doc_id, text, chunk_size, overlap))
    return out
