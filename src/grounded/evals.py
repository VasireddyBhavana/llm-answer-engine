"""Evaluation harness: measure retrieval and answers against a golden set, compare configs.

Retrieval metrics (cheap, no LLM):  hit_rate@k, recall@k, MRR
Answer metrics:                      correctness (keyword match), citation precision,
                                     faithfulness (LLM-as-judge), latency p50/p95
Run:  grounded eval            -> compares bm25 vs vector vs hybrid and writes reports/eval.md
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import LLM
from .rag import AnswerEngine

JUDGE_SYSTEM = """You are a strict JUDGE of faithfulness. Given a context and an answer, score \
from 0.0 to 1.0 the fraction of the answer's claims that are directly supported by the \
context. Respond with JSON only: {"score": <float>, "reason": "<one sentence>"}"""


@dataclass
class Example:
    id: str
    question: str
    relevant_docs: list[str]
    answer_keywords: list[str]


def load_golden(path: Path) -> list[Example]:
    return [Example(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- retrieval


def retrieval_metrics(engine: AnswerEngine, examples: list[Example], mode: str,
                      k: int) -> dict[str, float]:
    hits_at_k, recalls, rrs = [], [], []
    for ex in examples:
        docs = [h.chunk.doc_id for h in engine.retriever.search(ex.question, k=k, mode=mode)]
        relevant = set(ex.relevant_docs)
        hits_at_k.append(float(bool(relevant & set(docs))))
        recalls.append(len(relevant & set(docs)) / len(relevant))
        rank = next((i for i, d in enumerate(docs, start=1) if d in relevant), None)
        rrs.append(1.0 / rank if rank else 0.0)
    n = len(examples)
    return {"hit_rate": sum(hits_at_k) / n, "recall": sum(recalls) / n, "mrr": sum(rrs) / n}


# --------------------------------------------------------------------------- answers


def judge_faithfulness(judge: LLM, contexts: list[str], answer: str) -> float:
    prompt = f"<context>\n{chr(10).join(contexts)}\n</context>\n<answer>{answer}</answer>"
    raw = judge.complete(JUDGE_SYSTEM, [{"role": "user", "content": prompt}]).text
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        return max(0.0, min(1.0, float(json.loads(raw[start:end])["score"])))
    except (ValueError, KeyError, json.JSONDecodeError):
        return 0.0  # unparseable judge output counts as a failure, and is visible in the report


def answer_metrics(engine: AnswerEngine, examples: list[Example], mode: str,
                   judge: LLM | None = None) -> tuple[dict[str, float], list[dict[str, Any]]]:
    judge = judge or engine.llm
    rows, latencies = [], []
    for ex in examples:
        t0 = time.perf_counter()
        ans = engine.ask(ex.question, mode=mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        text = ans.text.lower()
        correct = float(all(kw.lower() in text for kw in ex.answer_keywords))
        cited_docs = [c.doc_id for c in ans.citations]
        cite_prec = (sum(d in ex.relevant_docs for d in cited_docs) / len(cited_docs)
                     if cited_docs else 0.0)
        faithful = judge_faithfulness(judge, ans.contexts, ans.text)
        rows.append({"id": ex.id, "question": ex.question, "answer": ans.text,
                     "cited": cited_docs, "correct": correct, "citation_precision": cite_prec,
                     "faithfulness": faithful})
    n = len(rows)
    lat = sorted(latencies)
    summary = {
        "correctness": sum(r["correct"] for r in rows) / n,
        "citation_precision": sum(r["citation_precision"] for r in rows) / n,
        "faithfulness": sum(r["faithfulness"] for r in rows) / n,
        "latency_p50_ms": statistics.median(lat),
        "latency_p95_ms": lat[min(n - 1, int(0.95 * n))],
    }
    return summary, rows


# --------------------------------------------------------------------------- report


def run_eval(engine: AnswerEngine, golden: Path, modes: list[str], k: int,
             out_dir: Path, with_answers: bool = True) -> dict[str, Any]:
    examples = load_golden(golden)
    report: dict[str, Any] = {"model": engine.llm.name, "embedder": engine.embedder.name,
                              "k": k, "n_examples": len(examples),
                              "chunk_size": engine.settings.chunk_size, "results": {}}
    for mode in modes:
        res: dict[str, Any] = {"retrieval": retrieval_metrics(engine, examples, mode, k)}
        if with_answers:
            res["answers"], res["rows"] = answer_metrics(engine, examples, mode)
        report["results"][mode] = res

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(report, indent=1))
    (out_dir / "eval.md").write_text(to_markdown(report))
    return report


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Eval report\n",
             f"model `{report['model']}` · embedder `{report['embedder']}` · "
             f"k={report['k']} · chunk_size={report['chunk_size']} · "
             f"{report['n_examples']} questions\n",
             "| retriever | hit@k | recall@k | MRR | correct | cite prec | faithful | p50 ms |",
             "|---|---|---|---|---|---|---|---|"]
    for mode, res in report["results"].items():
        r, a = res["retrieval"], res.get("answers", {})
        cells = [f"{r['hit_rate']:.2f}", f"{r['recall']:.2f}", f"{r['mrr']:.2f}"]
        for key in ("correctness", "citation_precision", "faithfulness"):
            cells.append(f"{a[key]:.2f}" if key in a else "–")
        cells.append(f"{a['latency_p50_ms']:.1f}" if a else "–")
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
