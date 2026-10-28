from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from grounded.api import create_app
from grounded.config import Settings
from grounded.evals import fake_integrations
from grounded.integrations import build_integration_tools
from grounded.llm import MockLLM
from grounded.rag import AnswerEngine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="mock",
        embedder="hashing",
        data_dir=ROOT / "data",
        db_path=tmp_path / "test.db",
        trace_file=tmp_path / "traces.jsonl",
        memory_file=tmp_path / "memory.json",
        jwt_secret="test-secret-that-is-long-enough-for-hs256-0123456789",
        admin_email="",
        rate_limit_per_min=1000,
    )


@pytest.fixture
def engine(settings: Settings) -> AnswerEngine:
    e = AnswerEngine(settings, llm=MockLLM())
    # Never touch the network in tests: route integrations to deterministic fakes.
    e.integration_tools = build_integration_tools(
        ["weather", "wikipedia", "webhook"], "https://hooks.eval.local/T000",
        transport=fake_integrations())
    yield e
    e.db.close()


@pytest.fixture
def users(engine: AnswerEngine) -> dict:
    a = engine.auth
    return {
        "admin": a.create_user("admin@corp.test", "admin-password", "Ada", "admin"),
        "hr": a.create_user("hr@corp.test", "hr-password-1", "Hana", "member", ["hr"]),
        "eng": a.create_user("eng@corp.test", "eng-password-1", "Eli", "member",
                             ["engineering"]),
        "viewer": a.create_user("viewer@corp.test", "viewer-password", "Vic", "viewer"),
    }


@pytest.fixture
def client(engine: AnswerEngine) -> TestClient:
    return TestClient(create_app(engine))


def login(client: TestClient, email: str, password: str) -> dict[str, str]:
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def headers(client: TestClient, users: dict) -> dict[str, dict[str, str]]:
    creds = {"admin": "admin-password", "hr": "hr-password-1", "eng": "eng-password-1",
             "viewer": "viewer-password"}
    return {name: login(client, users[name].email, pw) for name, pw in creds.items()}


def fake_http(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)
