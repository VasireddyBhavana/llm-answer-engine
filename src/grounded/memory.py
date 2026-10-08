"""Personalization via long-term memory.

Each user has a small profile: explicit preferences ("be concise", "I'm a beginner") and
implicit topic interest (which documents their past questions retrieved). Preferences change
*how* answers are written (injected into the system prompt); topic interest gently boosts
retrieval toward documents the user has engaged with. Users can inspect and delete everything.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Simple rule-based extraction. A production system would ask the LLM to extract memories,
# but rules are transparent, cheap, and never store something the user didn't say.
_PREFERENCE_PATTERNS = [
    (r"\b(be|keep it|answers?)\s+(short|brief|concise)\b", "style", "concise"),
    (r"\b(more detail|in depth|detailed|thorough)\b", "style", "detailed"),
    (r"\bi'?m (a )?(beginner|new to)\b", "level", "beginner"),
    (r"\bi'?m (an? )?(expert|senior|experienced)\b", "level", "expert"),
    (r"\b(use|give( me)?) (examples|analogies)\b", "examples", "yes"),
]

STYLE_TEXT = {
    ("style", "concise"): "Keep answers to 2-3 sentences.",
    ("style", "detailed"): "Give a thorough, detailed answer.",
    ("level", "beginner"): "The user is a beginner: avoid jargon and define terms.",
    ("level", "expert"): "The user is an expert: skip basics and be precise.",
    ("examples", "yes"): "Include a short concrete example.",
}


@dataclass
class UserProfile:
    user_id: str
    preferences: dict[str, str] = field(default_factory=dict)
    topic_counts: dict[str, int] = field(default_factory=dict)

    def system_addendum(self) -> str:
        lines = [STYLE_TEXT[(k, v)] for k, v in self.preferences.items() if (k, v) in STYLE_TEXT]
        return ("\nPersonalization (affects style only, never facts):\n- " + "\n- ".join(lines)
                if lines else "")

    def retrieval_boost(self, strength: float = 0.15) -> dict[str, float]:
        """Mild multiplicative boost for familiar topics, capped to avoid filter bubbles."""
        total = sum(self.topic_counts.values())
        if not total:
            return {}
        return {doc: 1.0 + strength * min(count / total, 0.5) * 2
                for doc, count in self.topic_counts.items()}


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}

    def get(self, user_id: str) -> UserProfile:
        raw = self._data.get(user_id)
        return UserProfile(**raw) if raw else UserProfile(user_id)

    def _put(self, profile: UserProfile) -> None:
        with self._lock:
            self._data[profile.user_id] = asdict(profile)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=1))

    def observe(self, user_id: str, message: str, retrieved_docs: list[str]) -> UserProfile:
        """Update a profile from one interaction. Returns the updated profile."""
        p = self.get(user_id)
        low = message.lower()
        for pattern, key, value in _PREFERENCE_PATTERNS:
            if re.search(pattern, low):
                p.preferences[key] = value
        counts = Counter(p.topic_counts)
        counts.update(set(retrieved_docs))
        p.topic_counts = dict(counts)
        self._put(p)
        return p

    def set_preference(self, user_id: str, key: str, value: str) -> UserProfile:
        p = self.get(user_id)
        p.preferences[key] = value
        self._put(p)
        return p

    def forget(self, user_id: str) -> None:
        with self._lock:
            self._data.pop(user_id, None)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=1))
