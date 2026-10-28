import json

from grounded.evals import (
    compare_to_baseline,
    flatten,
    judge_faithfulness,
    load_golden,
    retrieval_metrics,
    run_eval,
)
from grounded.llm import LLMResponse, MockLLM
from grounded.memory import MemoryStore
from grounded.retrieval import Retriever

# ---------------------------------------------------------------- memory


def test_preferences_extracted_and_injected(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    p = store.observe("u1", "I'm a beginner, please keep it short", ["rag"])
    assert p.preferences == {"level": "beginner", "style": "concise"}
    assert "beginner" in p.system_addendum()
    assert MemoryStore(tmp_path / "m.json").get("u1").topic_counts == {"rag": 1}


def test_boost_is_capped(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    for _ in range(50):
        store.observe("u1", "q", ["rag"])
    assert max(store.get("u1").retrieval_boost().values()) <= 1.15 + 1e-9


# ---------------------------------------------------------------- evals


def test_retrieval_metrics_on_golden_set(engine, settings):
    examples = load_golden(settings.data_dir / "eval" / "golden.jsonl")
    m = retrieval_metrics(engine, examples, "hybrid", k=4)
    assert m["recall"] >= 0.9 and 0 <= m["mrr"] <= 1


def test_judge_parses_scores_and_handles_garbage():
    assert judge_faithfulness(MockLLM(), ["the sky is blue"], "the sky is blue") == 1.0

    class Garbage:
        name = "g"

        def complete(self, *a, **k):
            return LLMResponse("not json")
    assert judge_faithfulness(Garbage(), ["x"], "y") == 0.0


def test_all_suites_run_and_pass(settings, tmp_path):
    report = run_eval(settings, ["rag", "security", "agent"], ["bm25", "hybrid"], 4, tmp_path)
    flat = flatten(report)
    assert flat["security.leak_rate"] == 0.0
    assert flat["security.authorized_recall"] == 1.0
    assert flat["agent.task_success"] == 1.0
    assert (tmp_path / "eval.md").read_text().startswith("# Eval report")


def test_security_eval_detects_a_broken_acl(settings, tmp_path, monkeypatch):
    """Mutation test: if access filtering is removed, the suite must report leaks."""
    original = Retriever.build_mask
    monkeypatch.setattr(Retriever, "build_mask",
                        lambda self, allow=None, tags=None, doc_ids=None:
                        original(self, None, tags, doc_ids))
    report = run_eval(settings, ["security"], [], 4, tmp_path)
    assert report["suites"]["security"]["metrics"]["leak_rate"] > 0
    assert compare_to_baseline(report, {"security.leak_rate": 0.0})


def test_baseline_gate(settings, tmp_path):
    report = run_eval(settings, ["agent"], [], 4, tmp_path)
    good = flatten(report)
    assert compare_to_baseline(report, good) == []
    stricter = {**good, "agent.task_success": 1.5, "agent.avg_steps": 0.1,
                "rag.hybrid.answers.latency_p50_ms": 0.0, "missing.metric": 1.0}
    problems = compare_to_baseline(report, stricter)
    assert len(problems) == 2  # task_success dropped, avg_steps rose; latency + missing ignored


def test_committed_baseline_matches_current_code(settings, tmp_path):
    """The CI gate uses evals/baseline.json; keep it honest."""
    from conftest import ROOT
    baseline = json.loads((ROOT / "evals" / "baseline.json").read_text())
    report = run_eval(settings, ["rag", "security", "agent"],
                      ["bm25", "vector", "hybrid", "hybrid+rerank"], 4, tmp_path)
    assert compare_to_baseline(report, baseline) == []
