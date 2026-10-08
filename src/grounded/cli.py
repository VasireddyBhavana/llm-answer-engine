"""Command line:  grounded ingest | ask | agent | eval | traces | memory"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent
from .evals import run_eval
from .rag import AnswerEngine


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="grounded", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="chunk + embed data/corpus into the index")

    a = sub.add_parser("ask", help="answer a question with citations (RAG)")
    a.add_argument("question")
    a.add_argument("--user")
    a.add_argument("--mode", choices=["bm25", "vector", "hybrid"])
    a.add_argument("--k", type=int)
    a.add_argument("--trace", action="store_true", help="print the timing trace")

    g = sub.add_parser("agent", help="run the tool-using agent on a task")
    g.add_argument("task")
    g.add_argument("--user")

    e = sub.add_parser("eval", help="run the golden-set evaluation")
    e.add_argument("--golden", default="data/eval/golden.jsonl")
    e.add_argument("--modes", default="bm25,vector,hybrid")
    e.add_argument("--k", type=int, default=4)
    e.add_argument("--out", default="reports")
    e.add_argument("--retrieval-only", action="store_true")
    e.add_argument("--min-recall", type=float, default=0.0,
                   help="exit non-zero if hybrid recall@k falls below this (for CI)")

    t = sub.add_parser("traces", help="show the most recent traces")
    t.add_argument("-n", type=int, default=5)

    m = sub.add_parser("memory", help="show or forget what is remembered about a user")
    m.add_argument("user")
    m.add_argument("--forget", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "traces":
        from .config import get_settings
        path = get_settings().trace_file
        lines = path.read_text().splitlines()[-args.n:] if path.exists() else []
        for line in lines:
            print(json.dumps(json.loads(line), indent=1))
        return 0

    engine = AnswerEngine()

    if args.cmd == "ingest":
        r = engine.build_index()
        docs = {c.doc_id for c in r.chunks}
        print(f"Indexed {len(r.chunks)} chunks from {len(docs)} documents "
              f"with {engine.embedder.name}.")
    elif args.cmd == "ask":
        ans = engine.ask(args.question, args.user, args.k, args.mode)
        print(ans.text + "\n")
        for c in ans.citations:
            print(f"  [{c.n}] {c.title} ({c.chunk_id})")
        if args.trace:
            print(json.dumps(ans.trace, indent=1))
    elif args.cmd == "agent":
        res = Agent(engine).run(args.task, args.user)
        for i, s in enumerate(res.steps, 1):
            print(f"step {i}: {s.tool}({json.dumps(s.arguments)}) -> "
                  f"{s.output[:90].replace(chr(10), ' ')}...")
        print(f"\n{res.answer}\n\n(stopped: {res.stopped_reason})")
    elif args.cmd == "eval":
        modes = args.modes.split(",")
        report = run_eval(engine, Path(args.golden), modes, args.k, Path(args.out),
                          with_answers=not args.retrieval_only)
        print((Path(args.out) / "eval.md").read_text())
        recall = report["results"].get("hybrid", {}).get("retrieval", {}).get("recall", 1.0)
        if recall < args.min_recall:
            print(f"FAIL: hybrid recall@{args.k} {recall:.2f} < {args.min_recall}")
            return 1
    elif args.cmd == "memory":
        if args.forget:
            engine.memory.forget(args.user)
            print(f"Forgot everything about {args.user}.")
        else:
            print(json.dumps(engine.memory.get(args.user).__dict__, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
