"""Command line for operators:  grounded <command> --help

  ingest        add files (PDF, DOCX, HTML, MD, TXT, CSV, JSON) with access groups and tags
  docs          list documents
  create-user   add a user (viewer | member | admin) with groups
  ask           answer a question with citations, optionally as a specific user
  agent         run the tool-using agent
  actions       list / approve / reject actions waiting for approval
  eval          run automated evaluation suites and the regression gate
  traces        show recent request traces
  audit         show the audit log
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .agent import Agent, ApprovalQueue
from .auth import Principal
from .config import get_settings
from .evals import compare_to_baseline, flatten, run_eval
from .rag import MODES, AnswerEngine


def _csv(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _as_user(engine: AnswerEngine, email: str | None) -> Principal:
    """CLI acts as the system (admin) unless --as names a user, which is handy for testing ACLs."""
    if not email:
        return Principal.system()
    row = engine.db.one("SELECT id FROM users WHERE email = ?", (email.lower(),))
    if not row:
        sys.exit(f"no user with email {email}")
    return engine.auth.get_user(row["id"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="grounded", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="add files or folders; with no paths, (re)seed data/corpus")
    i.add_argument("paths", nargs="*", type=Path)
    i.add_argument("--groups", default="everyone", help="comma-separated, e.g. hr,leadership")
    i.add_argument("--tags", default="")

    d = sub.add_parser("docs", help="list documents")
    d.add_argument("--as", dest="as_user")
    d.add_argument("--delete", metavar="DOC_ID")

    u = sub.add_parser("create-user", help="add a user")
    u.add_argument("email")
    u.add_argument("--role", default="member", choices=["viewer", "member", "admin"])
    u.add_argument("--groups", default="")
    u.add_argument("--name", default="")
    u.add_argument("--password", help="omit to be prompted (keeps it out of shell history)")

    a = sub.add_parser("ask", help="answer a question with citations")
    a.add_argument("question")
    a.add_argument("--as", dest="as_user", help="ask as this user's email (tests access control)")
    a.add_argument("--mode", choices=MODES)
    a.add_argument("--k", type=int)
    a.add_argument("--tags")
    a.add_argument("--trace", action="store_true", help="print the timing trace")

    g = sub.add_parser("agent", help="run the tool-using agent on a task")
    g.add_argument("task")
    g.add_argument("--as", dest="as_user")

    ac = sub.add_parser("actions", help="list or decide actions waiting for approval")
    ac.add_argument("--approve", metavar="ACTION_ID")
    ac.add_argument("--reject", metavar="ACTION_ID")

    e = sub.add_parser("eval", help="run evaluation suites")
    e.add_argument("--suites", default="rag,security,agent")
    e.add_argument("--modes", default="bm25,vector,hybrid,hybrid+rerank")
    e.add_argument("--k", type=int, default=4)
    e.add_argument("--out", default="reports")
    e.add_argument("--retrieval-only", action="store_true")
    e.add_argument("--check-baseline", metavar="FILE", type=Path,
                   help="exit non-zero if any metric regresses vs this file")
    e.add_argument("--update-baseline", metavar="FILE", type=Path,
                   help="write current metrics as the new baseline")
    e.add_argument("--tolerance", type=float, default=0.02)
    e.add_argument("--min-recall", type=float, default=0.0,
                   help="exit non-zero if hybrid recall@k falls below this")

    t = sub.add_parser("traces", help="show the most recent traces")
    t.add_argument("-n", type=int, default=5)

    au = sub.add_parser("audit", help="show the audit log")
    au.add_argument("-n", type=int, default=20)

    args = p.parse_args(argv)
    settings = get_settings()

    # Commands that don't need the engine -----------------------------------------
    if args.cmd == "traces":
        path = settings.trace_file
        for line in (path.read_text().splitlines()[-args.n:] if path.exists() else []):
            print(json.dumps(json.loads(line), indent=1))
        return 0

    if args.cmd == "eval":
        report = run_eval(settings, _csv(args.suites), _csv(args.modes), args.k, Path(args.out),
                          with_answers=not args.retrieval_only)
        print((Path(args.out) / "eval.md").read_text())
        code = 0
        recall = (report["suites"].get("rag", {}).get("hybrid", {})
                  .get("retrieval", {}).get("recall", 1.0))
        if recall < args.min_recall:
            print(f"FAIL: hybrid recall@{args.k} {recall:.2f} < {args.min_recall}")
            code = 1
        if args.check_baseline:
            problems = compare_to_baseline(report, json.loads(args.check_baseline.read_text()),
                                           args.tolerance)
            if problems:
                print("REGRESSIONS vs baseline:\n  " + "\n  ".join(problems))
                code = 1
            else:
                print(f"Baseline check passed ({args.check_baseline}).")
        if args.update_baseline:
            args.update_baseline.parent.mkdir(parents=True, exist_ok=True)
            args.update_baseline.write_text(json.dumps(flatten(report), indent=1, sort_keys=True)
                                            + "\n")
            print(f"Wrote baseline to {args.update_baseline}")
        return code

    engine = AnswerEngine(settings)

    if args.cmd == "ingest":
        if not args.paths:
            n = engine.seed_corpus()
            print(f"Seeded {n} new documents from {settings.data_dir / 'corpus'} "
                  f"({engine.store.count()} total).")
            return 0
        for path in args.paths:
            files = engine.store.add_folder(path, groups=_csv(args.groups), tags=_csv(args.tags)) \
                if path.is_dir() else [engine.store.add_path(path, groups=_csv(args.groups),
                                                             tags=_csv(args.tags))]
            for doc in files:
                status = "duplicate, skipped" if doc.duplicate else f"{doc.n_chunks} chunks"
                print(f"{doc.id}  {doc.filename}  [{', '.join(doc.groups)}]  {status}")
                engine.audit.record("system", "document.upload", doc.id, filename=doc.filename)
    elif args.cmd == "docs":
        if args.delete:
            print("deleted" if engine.store.delete(args.delete) else "not found")
            return 0
        for doc in engine.store.list(_as_user(engine, args.as_user)):
            print(f"{doc.id:16} {doc.title[:40]:40} groups={','.join(doc.groups):20} "
                  f"tags={','.join(doc.tags)}")
    elif args.cmd == "create-user":
        pw = args.password or getpass.getpass("Password (min 8 chars): ")
        user = engine.auth.create_user(args.email, pw, args.name, args.role, _csv(args.groups))
        print(f"Created {user.email} ({user.role}) groups={sorted(user.groups)} id={user.user_id}")
    elif args.cmd == "ask":
        ans = engine.ask(args.question, _as_user(engine, args.as_user), args.k, args.mode,
                         _csv(args.tags) or None)
        print(ans.text + "\n")
        for c in ans.citations:
            print(f"  [{c.n}] {c.title} — {c.section} ({c.chunk_id})")
        if args.trace:
            print(json.dumps(ans.trace, indent=1))
    elif args.cmd == "agent":
        res = Agent(engine, _as_user(engine, args.as_user)).run(args.task)
        for n, s in enumerate(res.steps, 1):
            print(f"step {n}: {s.tool}({json.dumps(s.arguments)}) [{s.status}] -> "
                  f"{s.output[:90].replace(chr(10), ' ')}")
        print(f"\n{res.answer}\n\n(stopped: {res.stopped_reason})")
        if res.pending_actions:
            print("Awaiting approval: " + ", ".join(res.pending_actions) +
                  "   ->  grounded actions --approve <id>")
    elif args.cmd == "actions":
        q, me = ApprovalQueue(engine), Principal.system()
        if args.approve or args.reject:
            print(json.dumps(q.decide(args.approve or args.reject, me, bool(args.approve)),
                             indent=1, default=str))
        else:
            for act in q.list(me):
                print(f"{act['id']}  {act['tool']}  {json.dumps(act['args'])}  by {act['user_id']}")
    elif args.cmd == "audit":
        for row in reversed(engine.audit.list(args.n)):
            print(json.dumps(asdict_safe(row), default=str))
    return 0


def asdict_safe(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "id"} if isinstance(row, dict) else asdict(row)


if __name__ == "__main__":
    sys.exit(main())
