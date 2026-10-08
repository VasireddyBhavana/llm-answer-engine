"""Minimal tracing: time each step of a request and append the trace to a JSONL file.

The same idea as LangSmith / Langfuse / OpenTelemetry spans, small enough to read in a minute.
Inspect traces with:  grounded traces   (or just open .traces/traces.jsonl)
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    name: str
    start: float
    duration_ms: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    kind: str
    input: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.time)
    spans: list[Span] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        s = Span(name, time.perf_counter(), attrs=dict(attrs))
        try:
            yield s
        finally:
            s.duration_ms = round((time.perf_counter() - s.start) * 1000, 2)
            self.spans.append(s)

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.id,
            "kind": self.kind,
            "total_ms": round(sum(s.duration_ms for s in self.spans), 2),
            "tokens": {"input": self.input_tokens, "output": self.output_tokens},
            "spans": [{"name": s.name, "ms": s.duration_ms, **s.attrs} for s in self.spans],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"ts": self.started, "input": self.input, **self.summary()}) + "\n")
