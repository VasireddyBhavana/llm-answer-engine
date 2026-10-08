"""Tiny text utilities shared by retrieval, the mock LLM and evals."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be because but by can do does for from has have how i if in into is it
    its of on or so such than that the their them then there these they this to was what when
    where which while who why will with you your we our not no more most also one each other
    about over just like very should would could""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_stem(t) for t in toks]


def _stem(t: str) -> str:
    """Very light suffix stripping so 'chunks' matches 'chunk'. Not a real stemmer."""
    for suf in ("ings", "ing", "ies", "es", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in _SENT.split(text) if s]
