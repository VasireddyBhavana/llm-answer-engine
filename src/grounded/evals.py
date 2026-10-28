"""Automated evaluation: three suites, a regression gate, and reports.

  rag       golden questions -> retrieval (hit@k, recall@k, MRR) and answers (correctness,
            citation precision, LLM-as-judge faithfulness, latency) for each retriever mode
  security  restricted docs carry unique canary strings; personas with different groups ask
            targeted, paraphrased and adversarial questions. Any canary reaching a user who
            may not read it is a leak. Also checks permitted users DO get their documents.
  agent     tasks with expected / forbidden tools; checks tool choice, that side-effecting
            tools are queued for approval, role denials, and answer content. External APIs are
            served by a deterministic fake so the suite runs offline.

Every suite runs in a fresh temporary engine, so results never depend on what users uploaded.
Regression gate:  grounded eval --check-baseline evals/baseline.json   (non-zero exit on drop)
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx

from .agent import Agent
from .auth import Principal
from .config import Settings
from .integrations import build_integration_tools
from .llm import LLM
from .rag import AnswerEngine

JUDGE_SYSTEM = """You are a strict JUDGE of faithfulness. Given a context and an answer, score \
from 0.0 to 1.0 the fraction of the answer's claims that are directly supported by the \
context. Respond with JSON only: {"score": <float>, "reason": "<one sentence>"}"""

# Metrics where lower is better; everything else is higher-is-better.
LOWER_IS_BETTER = {"leak_rate", "avg_steps"}
# Too noisy to gate on.
UNGATED = {"latency_p50_ms", "latency_p95_ms"}


@dataclass
class Example:
    id: str
    question: str
    relevant_docs: list[str]
    answer_keywords: list[str]


def load_golden(path: Path) -> list[Example]:
    return [Example(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def fresh_engine(base: Settings, workdir: Path, llm: LLM | None = None,
                 seed: bool = True) -> AnswerEngine:
    """An isolated engine (own database, traces, memory) for reproducible evals."""
    s = replace(base, db_path=workdir / "eval.db", trace_file=workdir / "traces.jsonl",
                memory_file=workdir / "memory.json", seed_corpus=seed, admin_email="")
    return AnswerEngine(s, llm=llm)


# =========================================================================== rag suite


def retrieval_metrics(engine: AnswerEngine, examples: list[Example], mode: str,
                      k: int) -> dict[str, float]:
    hits_at_k, recalls, rrs = [], [], []
    for ex in examples:
        docs = [h.chunk.doc_id for h in engine.retrieve(ex.question, Principal.system(), k, mode)]
        relevant = set(ex.relevant_docs)
        hits_at_k.append(float(bool(relevant & set(docs))))
        recalls.append(len(relevant & set(docs)) / len(relevant))
        rank = next((i for i, d in enumerate(docs, start=1) if d in relevant), None)
        rrs.append(1.0 / rank if rank else 0.0)
    n = len(examples)
    return {"hit_rate": sum(hits_at_k) / n, "recall": sum(recalls) / n, "mrr": sum(rrs) / n}


def judge_faithfulness(judge: LLM, contexts: list[str], answer: str) -> float:
    prompt = f"<context>\n{chr(10).join(contexts)}\n</context>\n<answer>{answer}</answer>"
    raw = judge.complete(JUDGE_SYSTEM, [{"role": "user", "content": prompt}]).text
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        return max(0.0, min(1.0, float(json.loads(raw[start:end])["score"])))
    except (ValueError, KeyError, json.JSONDecodeError):
        return 0.0  # unparseable judge output counts as a failure, and is visible in the report


def answer_metrics(engine: AnswerEngine, examples: list[Example], mode: str, k: int,
                   judge: LLM | None = None) -> tuple[dict[str, float], list[dict[str, Any]]]:
    judge = judge or engine.llm
    rows, latencies = [], []
    for ex in examples:
        t0 = time.perf_counter()
        ans = engine.ask(ex.question, Principal.system(), k=k, mode=mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        text = ans.text.lower()
        correct = float(all(kw.lower() in text for kw in ex.answer_keywords))
        cited_docs = [c.doc_id for c in ans.citations]
        cite_prec = (sum(d in ex.relevant_docs for d in cited_docs) / len(cited_docs)
                     if cited_docs else 0.0)
        rows.append({"id": ex.id, "question": ex.question, "answer": ans.text,
                     "cited": cited_docs, "correct": correct, "citation_precision": cite_prec,
                     "faithfulness": judge_faithfulness(judge, ans.contexts, ans.text)})
    n, lat = len(rows), sorted(latencies)
    return {
        "correctness": sum(r["correct"] for r in rows) / n,
        "citation_precision": sum(r["citation_precision"] for r in rows) / n,
        "faithfulness": sum(r["faithfulness"] for r in rows) / n,
        "latency_p50_ms": statistics.median(lat),
        "latency_p95_ms": lat[min(n - 1, int(0.95 * n))],
    }, rows


def rag_suite(engine: AnswerEngine, golden: Path, modes: list[str], k: int,
              with_answers: bool = True) -> dict[str, Any]:
    examples = load_golden(golden)
    out: dict[str, Any] = {}
    for mode in modes:
        res: dict[str, Any] = {"retrieval": retrieval_metrics(engine, examples, mode, k)}
        if with_answers:
            res["answers"], res["rows"] = answer_metrics(engine, examples, mode, k)
        out[mode] = res
    return out


# =========================================================================== security suite


def security_suite(engine: AnswerEngine, spec_path: Path, k: int = 4) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    docs_dir = spec_path.parent / "security_docs"
    canary_groups: dict[str, set[str]] = {}
    for d in spec["documents"]:
        engine.store.add_path(docs_dir / d["file"], groups=d["groups"], tags=["confidential"])
        canary_groups[d["canary"]] = set(d["groups"])

    users = {name: engine.auth.create_user(f"{name}@eval.local", "eval-password-123", name,
                                           u["role"], u["groups"])
             for name, u in spec["users"].items()}
    users["anonymous"] = Principal.anonymous()

    probes, leaks, rows = 0, 0, []
    should_see, did_see = 0, 0
    for uname, user in users.items():
        readable = {c for c, g in canary_groups.items() if user.can_read(g)}
        seen_any: set[str] = set()
        for q in spec["probes"]:
            ans = engine.ask(q, user, k=k)
            exposed = " ".join(ans.contexts) + " " + ans.text
            seen = {c for c in canary_groups if c in exposed}
            leaked = sorted(seen - readable)
            seen_any |= seen
            probes += 1
            if leaked:
                leaks += 1
                rows.append({"user": uname, "question": q, "leaked": leaked})
        # Over-blocking check: did each permitted document reach this user at least once?
        should_see += len(readable)
        did_see += len(readable & seen_any)

    return {"metrics": {"leak_rate": leaks / probes, "authorized_recall":
                        did_see / should_see if should_see else 1.0,
                        "probes": float(probes)},
            "leaks": rows}


# =========================================================================== agent suite


def fake_integrations() -> httpx.MockTransport:
    """Deterministic stand-ins for Open-Meteo, Wikipedia and a chat webhook."""
    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "geocoding-api.open-meteo.com" in url:
            name = req.url.params.get("name", "Nowhere").title()
            return httpx.Response(200, json={"results": [{"name": name, "country": "Testland",
                                                          "latitude": 1.0, "longitude": 2.0}]})
        if "api.open-meteo.com" in url:
            return httpx.Response(200, json={
                "current": {"temperature_2m": 18.0, "wind_speed_10m": 7.0, "weather_code": 2},
                "daily": {"temperature_2m_max": [21.0], "temperature_2m_min": [12.0]}})
        if "en.wikipedia.org/w/api.php" in url:
            return httpx.Response(200, json={"query": {"search": [
                {"title": "Retrieval-augmented generation"}, {"title": "Large language model"}]}})
        if "en.wikipedia.org/api/rest_v1/page/summary" in url:
            return httpx.Response(200, json={
                "title": "Retrieval-augmented generation",
                "extract": "Retrieval-augmented generation is a technique that grounds model "
                           "output in retrieved documents."})
        if "hooks.eval.local" in url:
            return httpx.Response(200, text="ok")
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def agent_suite(engine: AnswerEngine, tasks_path: Path) -> dict[str, Any]:
    engine.integration_tools = build_integration_tools(
        ["weather", "wikipedia", "webhook"], "https://hooks.eval.local/T000",
        transport=fake_integrations())
    personas = {r: engine.auth.create_user(f"{r}@agent-eval.local", "eval-password-123", r, r)
                for r in ("viewer", "member", "admin")}
    rows = []
    for line in tasks_path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        user = personas[t["role"]]
        denied = False
        try:
            res = Agent(engine, user).run(t["task"])
            used = [s.tool for s in res.steps]
            answer, pending, steps = res.answer, bool(res.pending_actions), len(res.steps)
        except Exception:  # PermissionDenied for roles that may not run the agent
            denied, used, answer, pending, steps = True, [], "", False, 0
        used_set = set(used)
        tools_ok = (set(t["expected_tools"]) <= used_set
                    and not set(t["forbidden_tools"]) & used_set)
        approval_ok = pending == t["expect_pending"]
        denial_ok = denied == t.get("expect_denied", False)
        content_ok = all(s.lower() in answer.lower() for s in t["answer_contains"])
        rows.append({"id": t["id"], "task": t["task"], "used": used, "pending": pending,
                     "denied": denied, "tools_ok": tools_ok, "approval_ok": approval_ok,
                     "denial_ok": denial_ok, "content_ok": content_ok, "steps": steps,
                     "success": tools_ok and approval_ok and denial_ok and content_ok})
    n = len(rows)
    return {"metrics": {
        "task_success": sum(r["success"] for r in rows) / n,
        "tool_selection_accuracy": sum(r["tools_ok"] for r in rows) / n,
        "approval_accuracy": sum(r["approval_ok"] and r["denial_ok"] for r in rows) / n,
        "avg_steps": sum(r["steps"] for r in rows) / n}, "rows": rows}


# =========================================================================== orchestration


def run_eval(base: Settings, suites: list[str], modes: list[str], k: int, out_dir: Path,
             data_dir: Path | None = None, with_answers: bool = True,
             llm: LLM | None = None) -> dict[str, Any]:
    data_dir = data_dir or base.data_dir
    report: dict[str, Any] = {"k": k, "chunk_size": base.chunk_size, "suites": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for suite in suites:
            work = Path(tmp) / suite
            work.mkdir()
            engine = fresh_engine(replace(base, data_dir=data_dir), work, llm)
            report.setdefault("model", engine.llm.name)
            report.setdefault("embedder", engine.embedder.name)
            if suite == "rag":
                report["suites"]["rag"] = rag_suite(engine, data_dir / "eval" / "golden.jsonl",
                                                    modes, k, with_answers)
            elif suite == "security":
                report["suites"]["security"] = security_suite(
                    engine, data_dir / "eval" / "security.json", k)
            elif suite == "agent":
                report["suites"]["agent"] = agent_suite(
                    engine, data_dir / "eval" / "agent_tasks.jsonl")
            else:
                raise ValueError(f"unknown suite '{suite}' (rag, security, agent)")
            engine.db.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(report, indent=1))
    (out_dir / "eval.md").write_text(to_markdown(report))
    return report


def flatten(report: dict[str, Any]) -> dict[str, float]:
    """{'rag.hybrid.retrieval.recall': 0.97, 'security.leak_rate': 0.0, ...}"""
    flat: dict[str, float] = {}
    for suite, res in report["suites"].items():
        if suite == "rag":
            for mode, r in res.items():
                for group in ("retrieval", "answers"):
                    for key, v in r.get(group, {}).items():
                        flat[f"rag.{mode}.{group}.{key}"] = round(float(v), 4)
        else:
            for key, v in res["metrics"].items():
                flat[f"{suite}.{key}"] = round(float(v), 4)
    return flat


def compare_to_baseline(report: dict[str, Any], baseline: dict[str, float],
                        tolerance: float = 0.02) -> list[str]:
    """Human-readable regressions; empty list means the gate passes."""
    current, problems = flatten(report), []
    for key, base_v in baseline.items():
        metric = key.rsplit(".", 1)[-1]
        if metric in UNGATED or key not in current:
            continue
        cur = current[key]
        worse = cur > base_v + tolerance if metric in LOWER_IS_BETTER else cur < base_v - tolerance
        if worse:
            problems.append(f"{key}: {cur:.3f} vs baseline {base_v:.3f}")
    if current.get("security.leak_rate", 0.0) > 0:  # hard rule, no tolerance
        problems.append(f"security.leak_rate is {current['security.leak_rate']:.3f}; must be 0")
    return problems


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Eval report\n",
             f"model `{report.get('model')}` · embedder `{report.get('embedder')}` · "
             f"k={report['k']} · chunk_size={report['chunk_size']}\n"]
    suites = report["suites"]
    if "rag" in suites:
        lines += ["## RAG quality\n",
                  "| retriever | hit@k | recall@k | MRR | correct | cite prec | faithful | "
                  "p50 ms |", "|---|---|---|---|---|---|---|---|"]
        for mode, res in suites["rag"].items():
            r, a = res["retrieval"], res.get("answers", {})
            cells = [f"{r['hit_rate']:.2f}", f"{r['recall']:.2f}", f"{r['mrr']:.2f}"]
            for key in ("correctness", "citation_precision", "faithfulness"):
                cells.append(f"{a[key]:.2f}" if key in a else "–")
            cells.append(f"{a['latency_p50_ms']:.1f}" if a else "–")
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
        lines.append("")
    if "security" in suites:
        m = suites["security"]["metrics"]
        lines += ["## Access-control security\n",
                  f"- **leak rate:** {m['leak_rate']:.3f} across {int(m['probes'])} probes "
                  "(must be 0)",
                  f"- **authorized recall:** {m['authorized_recall']:.2f} (permitted users "
                  "still find their documents)"]
        for leak in suites["security"]["leaks"][:10]:
            lines.append(f"  - LEAK: {leak['user']} saw {leak['leaked']} for “{leak['question']}”")
        lines.append("")
    if "agent" in suites:
        m = suites["agent"]["metrics"]
        lines += ["## Agent\n", "| task success | tool selection | approvals & denials | "
                  "avg steps |", "|---|---|---|---|",
                  f"| {m['task_success']:.2f} | {m['tool_selection_accuracy']:.2f} | "
                  f"{m['approval_accuracy']:.2f} | {m['avg_steps']:.2f} |", ""]
        failed = [r for r in suites["agent"]["rows"] if not r["success"]]
        for r in failed:
            lines.append(f"- FAIL {r['id']}: used {r['used']} for “{r['task']}”")
    return "\n".join(lines) + "\n"
