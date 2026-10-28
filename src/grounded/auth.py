"""Authentication and authorization.

* Passwords: scrypt (memory-hard, stdlib) with a random per-user salt; constant-time compare.
* Sessions: short-lived JWTs (HS256) carrying only the user id; role and groups are re-read from
  the database on every request, so disabling a user or changing a role takes effect at once.
* Machine access: API keys (`gk_...`). Only a SHA-256 hash is stored; the key is shown once.
* Authorization: three roles (viewer < member < admin) mapped to permissions, plus group
  membership for document-level access control (see `Principal.can_read`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field

import jwt

from .db import Database

ROLES = ("viewer", "member", "admin")

# What each role may do. Checked by the API and again inside the agent (defense in depth).
PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"ask", "read_docs"}),
    "member": frozenset({"ask", "read_docs", "agent", "upload_docs", "use_integrations"}),
    "admin": frozenset({"ask", "read_docs", "agent", "upload_docs", "use_integrations",
                        "manage_users", "manage_all_docs", "view_audit", "approve_any"}),
}

EVERYONE = "everyone"


class AuthError(Exception):
    """Credentials are missing, wrong or expired (HTTP 401)."""


class PermissionDenied(Exception):
    """The caller is authenticated but not allowed to do this (HTTP 403)."""


@dataclass(frozen=True)
class Principal:
    """Who is making a request. Passed to retrieval so access control happens in one place."""

    user_id: str
    email: str = ""
    name: str = ""
    role: str = "viewer"
    groups: frozenset[str] = field(default_factory=lambda: frozenset({EVERYONE}))

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def can(self, permission: str) -> bool:
        return permission in PERMISSIONS.get(self.role, frozenset())

    def require(self, permission: str) -> None:
        if not self.can(permission):
            raise PermissionDenied(f"role '{self.role}' lacks permission '{permission}'")

    def can_read(self, doc_groups: list[str] | tuple[str, ...] | frozenset[str]) -> bool:
        """A document is readable if the user shares at least one group with it."""
        return self.is_admin or bool(self.groups & set(doc_groups))

    @staticmethod
    def system() -> Principal:
        """Internal principal for CLI maintenance and evals. Never issued to HTTP clients."""
        return Principal("system", "system@local", "System", "admin", frozenset({EVERYONE}))

    @staticmethod
    def anonymous() -> Principal:
        return Principal("anonymous", role="viewer")


# --------------------------------------------------------------------------- passwords

_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)


# A real hash computed once, so failed logins for unknown emails take as long as wrong passwords
# (prevents discovering which emails have accounts by timing the response).
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password")


# --------------------------------------------------------------------------- service


class AuthService:
    def __init__(self, db: Database, jwt_secret: str = "", ttl_minutes: int = 60):
        self.db = db
        self.ttl = ttl_minutes * 60
        self.ephemeral_secret = not jwt_secret
        self.secret = jwt_secret or secrets.token_urlsafe(48)

    # -- users ---------------------------------------------------------------
    def create_user(self, email: str, password: str, name: str = "", role: str = "member",
                    groups: list[str] | None = None) -> Principal:
        email = email.strip().lower()
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if "@" not in email:
            raise ValueError("invalid email")
        if self.db.one("SELECT 1 FROM users WHERE email = ?", (email,)):
            raise ValueError("a user with this email already exists")
        groups = sorted({EVERYONE, *(g.strip().lower() for g in (groups or []) if g.strip())})
        uid = "u_" + uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO users(id, email, name, role, groups_json, password_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, email, name or email.split("@")[0], role, json.dumps(groups),
             hash_password(password), time.time()))
        return self.get_user(uid)

    def get_user(self, user_id: str) -> Principal:
        row = self.db.one("SELECT * FROM users WHERE id = ? AND disabled = 0", (user_id,))
        if not row:
            raise AuthError("user not found or disabled")
        return Principal(row["id"], row["email"], row["name"], row["role"],
                         frozenset(json.loads(row["groups_json"])))

    def list_users(self) -> list[Principal]:
        rows = self.db.query("SELECT id FROM users WHERE disabled = 0 ORDER BY created_at")
        return [self.get_user(r["id"]) for r in rows]

    def update_user(self, user_id: str, role: str | None = None,
                    groups: list[str] | None = None, disabled: bool | None = None) -> None:
        if role is not None:
            if role not in ROLES:
                raise ValueError(f"role must be one of {ROLES}")
            self.db.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        if groups is not None:
            clean = sorted({EVERYONE, *(g.strip().lower() for g in groups if g.strip())})
            self.db.execute("UPDATE users SET groups_json = ? WHERE id = ?",
                            (json.dumps(clean), user_id))
        if disabled is not None:
            self.db.execute("UPDATE users SET disabled = ? WHERE id = ?", (int(disabled), user_id))

    def count_users(self) -> int:
        return self.db.one("SELECT COUNT(*) AS n FROM users")["n"]

    # -- login + tokens --------------------------------------------------------
    def authenticate(self, email: str, password: str) -> Principal:
        row = self.db.one("SELECT * FROM users WHERE email = ? AND disabled = 0",
                          (email.strip().lower(),))
        ok = verify_password(password, row["password_hash"] if row else _DUMMY_HASH)
        if not row or not ok:
            raise AuthError("invalid email or password")
        return self.get_user(row["id"])

    def issue_token(self, principal: Principal) -> str:
        now = int(time.time())
        return jwt.encode({"sub": principal.user_id, "iat": now, "exp": now + self.ttl,
                           "typ": "access"}, self.secret, algorithm="HS256")

    def verify_token(self, token: str) -> Principal:
        try:
            claims = jwt.decode(token, self.secret, algorithms=["HS256"],
                                options={"require": ["sub", "exp", "iat"]})
        except jwt.ExpiredSignatureError as e:
            raise AuthError("token expired") from e
        except jwt.PyJWTError as e:
            raise AuthError("invalid token") from e
        if claims.get("typ") != "access":
            raise AuthError("invalid token type")
        return self.get_user(claims["sub"])

    # -- API keys --------------------------------------------------------------
    def create_api_key(self, principal: Principal, name: str) -> tuple[str, str]:
        """Returns (key_id, plaintext_key). The plaintext is never stored."""
        raw = "gk_" + secrets.token_urlsafe(32)
        kid = "k_" + uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO api_keys(id, user_id, name, key_hash, prefix, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (kid, principal.user_id, name, _sha256(raw), raw[:10], time.time()))
        return kid, raw

    def verify_api_key(self, raw: str) -> Principal:
        row = self.db.one("SELECT * FROM api_keys WHERE key_hash = ? AND revoked = 0",
                          (_sha256(raw),))
        if not row:
            raise AuthError("invalid API key")
        self.db.execute("UPDATE api_keys SET last_used = ? WHERE id = ?", (time.time(), row["id"]))
        return self.get_user(row["user_id"])

    def list_api_keys(self, principal: Principal) -> list[dict]:
        rows = self.db.query("SELECT id, name, prefix, created_at, last_used, revoked "
                             "FROM api_keys WHERE user_id = ?", (principal.user_id,))
        return [dict(r) for r in rows]

    def revoke_api_key(self, principal: Principal, key_id: str) -> bool:
        return self.db.execute("UPDATE api_keys SET revoked = 1 WHERE id = ? AND user_id = ?",
                               (key_id, principal.user_id)) > 0

    def authenticate_bearer(self, credential: str) -> Principal:
        """Accepts either a JWT or an API key in `Authorization: Bearer <credential>`."""
        if credential.startswith("gk_"):
            return self.verify_api_key(credential)
        return self.verify_token(credential)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()
