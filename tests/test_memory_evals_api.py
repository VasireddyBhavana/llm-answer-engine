from fastapi.testclient import TestClient

from grounded import api
from grounded.agent import Agent
from grounded.evals import judge_faithfulness, load_golden, retrieval_metrics, run_eval
from grounded.llm import MockLLM
from grounded.memory import MemoryStore

# ---------------------------------------------------------------- memory


def test_preferences_extracted_and_injected(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    p = store.observe("u1", "I'm a beginner, please keep it short", ["rag"])
    assert p.preferences == {"level": "beginner", "style": "concise"}
    assert "beginner" in p.system_addendum()
    assert MemoryStore(tmp_path / "m.json").get("u1").topic_counts == {"rag": 1}  # persisted


def test_boost_is_capped(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    for _ in range(50):
        store.observe("u1", "q", ["rag"])
    assert max(store.get("u1").retrieval_boost().values()) <= 1.15 + 1e-9


def test_forget(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    store.observe("u1", "be concise", [])
    store.forget("u1")
    assert store.get("u1").preferences == {}


def test_agent_can_remember_preference(engine):
    agent = Agent(engine)
    agent.user_id = "u2"
    agent._run_tool("remember_preference", {"key": "level", "value": "expert"})
    assert engine.memory.get("u2").preferences["level"] == "expert"


# ---------------------------------------------------------------- evals


def test_retrieval_metrics_on_golden_set(engine, settings):
    examples = load_golden(settings.data_dir / "eval" / "golden.jsonl")
    m = retrieval_metrics(engine, examples, "hybrid", k=4)
    assert m["recall"] >= 0.8  # regression guard: fail CI if retrieval gets worse
    assert 0 <= m["mrr"] <= 1


def test_judge_parses_scores_and_handles_garbage():
    assert judge_faithfulness(MockLLM(), ["the sky is blue"], "the sky is blue") == 1.0

    class Garbage:
        name = "g"
        def complete(self, *a, **k):
            from grounded.llm import LLMResponse
            return LLMResponse("not json")
    assert judge_faithfulness(Garbage(), ["x"], "y") == 0.0


def test_run_eval_writes_reports(engine, settings, tmp_path):
    report = run_eval(engine, settings.data_dir / "eval" / "golden.jsonl", ["bm25", "hybrid"],
                      k=4, out_dir=tmp_path)
    assert set(report["results"]) == {"bm25", "hybrid"}
    assert (tmp_path / "eval.md").read_text().startswith("# Eval report")


# ---------------------------------------------------------------- api


def test_api_endpoints(engine, monkeypatch):
    monkeypatch.setattr(api, "engine", lambda: engine)
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"

    r = client.post("/ask", json={"question": "What is nucleus sampling?", "user_id": "u3"})
    assert r.status_code == 200 and r.json()["citations"]

    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "mode": "magic"}).status_code == 422

    assert client.post("/agent", json={"task": "What is 6*7?"}).json()["answer"].count("42")
    assert client.get("/users/u3/memory").json()["topic_counts"]
    assert client.delete("/users/u3/memory").status_code == 200
    assert client.delete("/users/u3/memory").status_code == 404
