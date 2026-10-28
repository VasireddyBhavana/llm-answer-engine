import time

import jwt
import pytest

from grounded.auth import (
    AuthError,
    AuthService,
    PermissionDenied,
    Principal,
    hash_password,
    verify_password,
)
from grounded.db import Database


@pytest.fixture
def auth() -> AuthService:
    return AuthService(Database(":memory:"), "s" * 48, ttl_minutes=1)


def test_password_hashing_is_salted_and_verifiable():
    h1, h2 = hash_password("correct horse"), hash_password("correct horse")
    assert h1 != h2 and h1.startswith("scrypt$")
    assert verify_password("correct horse", h1)
    assert not verify_password("wrong horse!", h1)
    assert not verify_password("anything", "garbage")


def test_short_passwords_rejected(auth):
    with pytest.raises(ValueError):
        auth.create_user("a@b.co", "short")


def test_login_token_roundtrip_and_failures(auth):
    u = auth.create_user("Ann@Corp.test", "password-123", role="member", groups=["HR", " "])
    assert u.groups == {"everyone", "hr"}  # normalised, everyone always included
    assert auth.authenticate("ann@corp.test", "password-123") == u
    with pytest.raises(AuthError):
        auth.authenticate("ann@corp.test", "nope-nope")
    with pytest.raises(AuthError):
        auth.authenticate("nobody@corp.test", "password-123")
    assert auth.verify_token(auth.issue_token(u)) == u


def test_tampered_expired_and_foreign_tokens_rejected(auth):
    u = auth.create_user("t@corp.test", "password-123")
    token = auth.issue_token(u)
    with pytest.raises(AuthError):
        auth.verify_token(token[:-2] + ("A" if token[-1] != "A" else "B") + token[-1])
    expired = jwt.encode({"sub": u.user_id, "iat": 0, "exp": int(time.time()) - 5,
                          "typ": "access"}, auth.secret, algorithm="HS256")
    with pytest.raises(AuthError, match="expired"):
        auth.verify_token(expired)
    forged = jwt.encode({"sub": u.user_id, "iat": 0, "exp": int(time.time()) + 60,
                         "typ": "access"}, "attacker-secret-" * 4, algorithm="HS256")
    with pytest.raises(AuthError):
        auth.verify_token(forged)


def test_disabled_user_loses_access_immediately(auth):
    u = auth.create_user("d@corp.test", "password-123")
    token = auth.issue_token(u)
    auth.update_user(u.user_id, disabled=True)
    with pytest.raises(AuthError):
        auth.verify_token(token)


def test_role_change_applies_to_existing_tokens(auth):
    u = auth.create_user("r@corp.test", "password-123", role="viewer")
    token = auth.issue_token(u)
    auth.update_user(u.user_id, role="admin")
    assert auth.verify_token(token).is_admin


def test_api_keys_hashed_listed_and_revocable(auth):
    u = auth.create_user("k@corp.test", "password-123")
    kid, raw = auth.create_api_key(u, "ci")
    assert raw.startswith("gk_")
    assert auth.authenticate_bearer(raw) == u
    stored = auth.db.one("SELECT key_hash FROM api_keys WHERE id = ?", (kid,))["key_hash"]
    assert raw not in stored
    assert auth.list_api_keys(u)[0]["prefix"] == raw[:10]
    assert auth.revoke_api_key(u, kid)
    with pytest.raises(AuthError):
        auth.authenticate_bearer(raw)


def test_permissions_and_group_reads():
    viewer = Principal("v", role="viewer")
    member = Principal("m", role="member", groups=frozenset({"everyone", "hr"}))
    assert viewer.can("ask") and not viewer.can("agent")
    assert member.can("upload_docs") and not member.can("manage_users")
    with pytest.raises(PermissionDenied):
        viewer.require("agent")
    assert member.can_read(["hr"]) and not member.can_read(["finance"])
    assert Principal.system().can_read(["anything"])
