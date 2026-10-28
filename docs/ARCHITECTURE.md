# Architecture

## Request lifecycle: `POST /ask`

1. **Authenticate.** `Authorization: Bearer <credential>`. A `gk_…` prefix means API key (looked up by SHA-256 hash); anything else is verified as an HS256 JWT. Either way the user row is re-read, so disabled users and role changes take effect immediately.
2. **Rate limit.** Sliding 60-second window per user (and per email on `/auth/login`).
3. **Authorize.** The role must hold the `ask` permission.
4. **Retrieve as the user.** `AnswerEngine.retrieve(question, principal)` builds a boolean mask over all chunks: readable by the user (shares a group with the document) AND matching any tag/document filters. BM25 and vector search both apply the mask before ranking; RRF fuses them; `hybrid+rerank` asks the LLM to rescore the top 12.
5. **Generate.** Numbered `<source>` blocks plus a system prompt that requires citations and treats source text as data.
6. **Validate citations.** `[n]` markers that don't map to a retrieved source are removed.
7. **Record.** Trace (span timings, token usage) to `traces.jsonl`; audit event to SQLite; personalization memory updated.

## Request lifecycle: `POST /agent`

1. Authenticate + require `agent` (members and admins).
2. `build_tools(engine, principal)` returns only the tools the role may use. Built-ins (`search_knowledge_base`, `list_documents`, `calculator`, `remember_preference`) are bound to the caller, so knowledge-base search inside the agent is filtered exactly like `/ask`.
3. Loop (max `GROUNDED_MAX_AGENT_STEPS`): the model sees the task and tool schemas, emits tool calls, the app runs them, results go back wrapped in `<tool_output untrusted="true">`.
4. At execution each call is checked again (unknown tool → `denied`; permission → `denied`), audited, and errors are returned to the model as text rather than raised.
5. Tools with `requires_approval` (currently `post_message`) are never executed in the loop. A row is written to `pending_actions` and the model is told the action is waiting for approval.

## Approval flow

```
agent -> pending_actions(status=pending)
      -> POST /actions/{id}/decision {approve: true}   (requester or admin; others get 404)
      -> re-load requester, rebuild their tools (permission re-check)
      -> run tool -> UPDATE ... WHERE status='pending'  (rowcount 0 => someone else decided)
      -> audit action.approved | action.rejected | action.failed
```

## Security model

| Threat | Mitigation |
|---|---|
| User reads documents outside their groups | Group ACL on every document; mask applied before ranking; agent search bound to caller; 404 on direct access; security eval with canaries in CI (zero tolerance) |
| Stolen/forged token | HS256 with a server secret; short TTL; `exp`/`iat` required; signature checked; user re-validated each request |
| Password guessing | scrypt hashing, per-email login rate limit, timing-equalised failures, failures audited |
| Leaked API key | Only hash stored; prefix shown for identification; revocable; `last_used` tracked |
| Privilege escalation via uploads | Members can share only into groups they belong to; only owner/admin can delete |
| Prompt injection from documents/tool output | Sources and tool outputs labelled as untrusted data; least-privilege tools; side effects need a human; citations validated |
| Agent calling internal services (SSRF) | `SafeHTTP` host allowlist, https only, no redirects |
| Slow or failing third-party APIs | Timeouts, bounded retries with backoff, `Retry-After`, response-size cap; errors returned to the model |
| Runaway agent | Step limit; approval for side effects |
| Unaccountable actions | Audit log of logins, uploads, deletes, questions, tool calls, approvals, denials |

Known gaps (deliberate, for a project this size): no SSO/OIDC, no per-tenant encryption, in-memory rate limiter (use Redis across replicas), single shared JWT secret (no rotation), memory store is a JSON file, prompt-injection defences are hygiene rather than guarantees.

## Data model (SQLite)

`users` (role, groups, scrypt hash) · `api_keys` (hash, prefix, revoked) · `documents` (groups, tags, sha256, owner) · `chunks` (text, section, float32 vector BLOB) · `audit_log` · `pending_actions` · `meta` (schema version, embedder name).

Vectors are stored with their documents so the in-memory index can be rebuilt on start. If the configured embedder changes, all chunks are re-embedded automatically, because vectors from different models aren't comparable.

## Scaling notes

| Today | At scale |
|---|---|
| Exact cosine over a numpy matrix, rebuilt after writes | ANN index (HNSW via pgvector / Qdrant), incremental upserts |
| BM25 in Python | OpenSearch / Elasticsearch / Tantivy |
| Mask built per query | Push ACL filters into the index (metadata filter on group ids) |
| Synchronous ingestion | Queue + workers; OCR for scanned PDFs |
| SQLite | Postgres; audit log to an append-only store |
| In-process rate limiter | Redis |
| JSONL traces | OpenTelemetry → Langfuse / Honeycomb / Datadog |

## Evaluation design

All suites build a fresh engine in a temp directory, so results don't depend on what anyone uploaded.

- **rag**: retrieval metrics (hit@k, recall@k, MRR) are scored separately from answer metrics (keyword correctness, citation precision, LLM-judge faithfulness), because most RAG failures are retrieval failures.
- **security**: `leak_rate` = fraction of (persona, probe) pairs where any forbidden canary appears in retrieved context or the answer. `authorized_recall` = fraction of permitted documents each persona reached at least once, which catches over-blocking. `tests/test_evals_memory.py::test_security_eval_detects_a_broken_acl` disables the filter and asserts leaks are reported, so the eval itself is tested.
- **agent**: per-task expected/forbidden tools, expected approval, expected denial, answer content. External APIs are served by `fake_integrations()` so results are deterministic.
- **gate**: `compare_to_baseline` flags any gated metric worse than baseline by more than the tolerance (lower-is-better for `leak_rate` and `avg_steps`; latency ignored), plus a hard rule that `leak_rate` must be 0.
