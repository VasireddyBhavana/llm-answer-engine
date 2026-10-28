"""File parsers: turn uploaded bytes into markdown-ish text the chunker understands.

Each loader emits headings (`#`, `##`) where the source has structure — PDF pages, DOCX
heading styles, HTML <h1-6> — so chunks never straddle sections and citations can point at
"Page 3" or "Refund policy" instead of an anonymous text window.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import PurePath

SUPPORTED = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html", ".htm": "text/html",
    ".md": "text/markdown", ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
}


class UnsupportedFile(ValueError):
    pass


@dataclass
class ParsedDocument:
    title: str
    text: str       # markdown-like, with headings marking structure
    mime: str
    meta: dict = field(default_factory=dict)


def parse_file(filename: str, data: bytes) -> ParsedDocument:
    suffix = PurePath(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedFile(f"unsupported file type '{suffix or filename}'. "
                              f"Supported: {', '.join(sorted(SUPPORTED))}")
    stem = PurePath(filename).stem.replace("_", " ").replace("-", " ").strip() or "Untitled"
    parser = {".pdf": _pdf, ".docx": _docx, ".html": _html, ".htm": _html, ".csv": _csv,
              ".json": _json}.get(suffix, _text)
    doc = parser(data, stem)
    doc.mime = SUPPORTED[suffix]
    doc.text = _normalise(doc.text)
    if not doc.text.strip():
        raise UnsupportedFile("no extractable text (scanned PDFs need OCR first)")
    if not re.match(r"^# ", doc.text):  # ensure a title heading for the chunker
        doc.text = f"# {doc.title}\n\n{doc.text}"
    return doc


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


# --------------------------------------------------------------------------- per-format


def _text(data: bytes, stem: str) -> ParsedDocument:
    text = _decode(data)
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return ParsedDocument(m.group(1).strip() if m else stem, text, "")


def _pdf(data: bytes, stem: str) -> ParsedDocument:
    from pypdf import PdfReader  # imported lazily: optional heavy dependency

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise UnsupportedFile("encrypted PDFs are not supported")
    info = reader.metadata or {}
    title = (getattr(info, "title", None) or "").strip() or stem
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if txt:
            # Re-flow hard line breaks inside paragraphs; keep blank-line paragraph breaks.
            txt = re.sub(r"(?<![.!?:])\n(?!\n)", " ", txt)
            pages.append(f"## Page {i}\n\n{txt}")
    return ParsedDocument(title, f"# {title}\n\n" + "\n\n".join(pages), "",
                          {"pages": len(reader.pages)})


def _docx(data: bytes, stem: str) -> ParsedDocument:
    import docx  # python-docx

    d = docx.Document(io.BytesIO(data))
    title = (d.core_properties.title or "").strip()
    lines: list[str] = []
    for p in d.paragraphs:
        txt = p.text.strip()
        if not txt:
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style == "title":
            title = title or txt
            lines.append(f"# {txt}")
        elif style.startswith("heading"):
            level = int(re.sub(r"\D", "", style) or 1)
            lines.append(f"{'#' * min(level + 1, 6)} {txt}")
        else:
            lines.append(txt)
        lines.append("")
    for t_i, table in enumerate(d.tables, start=1):
        lines.append(f"## Table {t_i}\n")
        rows = [[c.text.strip() for c in r.cells] for r in table.rows]
        if rows:
            header = rows[0]
            for r in rows[1:]:
                lines.append("; ".join(f"{h}: {v}" for h, v in zip(header, r, strict=False)))
                lines.append("")
    return ParsedDocument(title or stem, "\n".join(lines), "")


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form"}
    BLOCK = {"p", "div", "li", "tr", "br", "section", "article", "table", "ul", "ol", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False
        self._heading: int | None = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif re.fullmatch(r"h[1-6]", tag):
            self._heading = int(tag[1])
            self.out.append("\n\n" + "#" * self._heading + " ")
        elif tag in self.BLOCK:
            self.out.append("\n\n" if tag != "li" else "\n- ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self._in_title = False
        elif re.fullmatch(r"h[1-6]", tag):
            self._heading = None
            self.out.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.out.append(re.sub(r"\s+", " ", data))


def _html(data: bytes, stem: str) -> ParsedDocument:
    p = _HTMLText()
    p.feed(_decode(data))
    text = "".join(p.out)
    text = re.sub(r" *\n *", "\n", text)
    title = p.title.strip() or stem
    if not re.search(r"^# ", text.strip()):
        text = f"# {title}\n\n{text}"
    return ParsedDocument(title, text, "")


def _csv(data: bytes, stem: str) -> ParsedDocument:
    """Each row becomes a 'column: value' paragraph, so rows are retrievable individually."""
    rows = list(csv.reader(io.StringIO(_decode(data))))
    if not rows:
        return ParsedDocument(stem, "", "")
    header, body = rows[0], rows[1:]
    paras = ["; ".join(f"{h.strip()}: {v.strip()}" for h, v in zip(header, r, strict=False)
                       if v.strip()) for r in body]
    return ParsedDocument(stem, f"# {stem}\n\nColumns: {', '.join(header)}\n\n" +
                          "\n\n".join(p for p in paras if p), "", {"rows": len(body)})


def _json(data: bytes, stem: str) -> ParsedDocument:
    obj = json.loads(_decode(data))
    lines: list[str] = []

    def walk(o, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        else:
            lines.append(f"{path}: {o}")

    walk(obj, "")
    return ParsedDocument(stem, "\n\n".join(lines), "")
