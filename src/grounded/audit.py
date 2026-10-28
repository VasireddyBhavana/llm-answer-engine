"""Append-only audit log: who did what, to which resource, and whether it was allowed.

Logged: logins (success and failure), document uploads/deletes, questions, agent runs, every
tool call, approvals, and permission denials. Question text is truncated and nothing secret
(passwords, keys, tokens) is ever passed in.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .db import Database


class AuditLog:
    def __init__(self, db: Database):
        self.db = db

    def record(self, user_id: str | None, action: str, resource: str | None = None,
               outcome: str = "ok", **details: Any) -> None:
        self.db.execute(
            "INSERT INTO audit_log(ts, user_id, action, resource, outcome, details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), user_id, action, resource, outcome,
             json.dumps(details, default=str)[:4000]))

    def list(self, limit: int = 100, user_id: str | None = None,
             action: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM audit_log WHERE 1=1", []
        if user_id:
            sql, params = sql + " AND user_id = ?", [*params, user_id]
        if action:
            sql, params = sql + " AND action = ?", [*params, action]
        rows = self.db.query(sql + " ORDER BY id DESC LIMIT ?", (*params, min(limit, 1000)))
        return [{**dict(r), "details": json.loads(r["details"])} for r in rows]
