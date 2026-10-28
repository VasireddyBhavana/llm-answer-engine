"""HTTP API.

Run:   uvicorn grounded.api:app --reload      then open http://localhost:8000/docs

Auth:  POST /auth/login with email + password -> access token (JWT)
       send `Authorization: Bearer <token>` (or an API key `gk_...`) on every other call.
"""


import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .agent import Agent, ApprovalQueue, build_tools
from .auth import ROLES, AuthError, PermissionDenied, Principal
from .docstore import DocumentTooLarge
from .loaders import UnsupportedFile
from .rag import MODES, AnswerEngine

bearer = HTTPBearer(auto_error=False)


# --------------------------------------------------------------------------- rate limiting


class RateLimiter:
    """Sliding-window limit per user. In-memory: use Redis when running several replicas."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise HTTPException(429, "rate limit exceeded; try again shortly",
                                    headers={"Retry-After": str(int(60 - (now - q[0])) + 1)})
            q.append(now)


# --------------------------------------------------------------------------- schemas


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateUserRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    name: str = ""
    role: str = Field(default="member", pattern=f"^({'|'.join(ROLES)})$")
    groups: list[str] = []


class UpdateUserRequest(BaseModel):
    role: str | None = Field(default=None, pattern=f"^({'|'.join(ROLES)})$")
    groups: list[str] | None = None
    disabled: bool | None = None


class ApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=10)
    mode: str | None = Field(default=None, pattern="^(" + "|".join(
        m.replace("+", r"\+") for m in MODES) + ")$")
    tags: list[str] | None = None
    doc_ids: list[str] | None = None


class AgentRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)


class DecisionRequest(BaseModel):
    approve: bool


# --------------------------------------------------------------------------- app


def create_app(engine: AnswerEngine | None = None) -> FastAPI:
    app = FastAPI(title="LLM Answer Engine",
                  description="Enterprise assistant: secure document ingestion, hybrid "
                              "retrieval, tool calling, integrations and evaluation.",
                  version="0.2.0")
    holder: dict[str, Any] = {"engine": engine}
    lock = threading.Lock()

    def get_engine() -> AnswerEngine:
        with lock:
            if holder["engine"] is None:
                holder["engine"] = AnswerEngine()
            return holder["engine"]

    limiter_holder: dict[str, RateLimiter] = {}

    def limiter() -> RateLimiter:
        if "l" not in limiter_holder:
            limiter_holder["l"] = RateLimiter(get_engine().settings.rate_limit_per_min)
        return limiter_holder["l"]

    @app.exception_handler(AuthError)
    def _auth_error(_: Request, exc: AuthError):
        return JSONResponse({"detail": str(exc)}, 401, headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(PermissionDenied)
    def _denied(_: Request, exc: PermissionDenied):
        return JSONResponse({"detail": str(exc)}, 403)

    def current_user(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> Principal:
        if creds is None:
            raise AuthError("missing bearer token")
        principal = get_engine().auth.authenticate_bearer(creds.credentials)
        limiter().check(principal.user_id)
        return principal

    User = Annotated[Principal, Depends(current_user)]

    def require(p: Principal, permission: str) -> None:
        if not p.can(permission):
            get_engine().audit.record(p.user_id, "permission.denied", permission, "denied")
            raise PermissionDenied(f"role '{p.role}' cannot {permission.replace('_', ' ')}")

    # ---------------------------------------------------------------- public
    @app.get("/health", tags=["system"])
    def health() -> dict:
        e = get_engine()
        return {"status": "ok", "model": e.llm.name, "embedder": e.embedder.name,
                "documents": e.store.count(), "chunks": len(e.retriever.chunks)}

    # ---------------------------------------------------------------- auth
    @app.post("/auth/login", tags=["auth"])
    def login(req: LoginRequest) -> dict:
        e = get_engine()
        limiter().check(f"login:{req.email.lower()}")  # slows password guessing
        try:
            p = e.auth.authenticate(req.email, req.password)
        except AuthError:
            e.audit.record(None, "auth.login", req.email.lower()[:120], "denied")
            raise
        e.audit.record(p.user_id, "auth.login")
        return {"access_token": e.auth.issue_token(p), "token_type": "bearer",
                "expires_in": e.auth.ttl}

    @app.get("/auth/me", tags=["auth"])
    def me(user: User) -> dict:
        return _principal_dict(user)

    @app.post("/auth/api-keys", tags=["auth"], status_code=201)
    def create_key(req: ApiKeyRequest, user: User) -> dict:
        kid, raw = get_engine().auth.create_api_key(user, req.name)
        get_engine().audit.record(user.user_id, "apikey.create", kid)
        return {"id": kid, "api_key": raw, "note": "Store this key now; it is not shown again."}

    @app.get("/auth/api-keys", tags=["auth"])
    def list_keys(user: User) -> list[dict]:
        return get_engine().auth.list_api_keys(user)

    @app.delete("/auth/api-keys/{key_id}", tags=["auth"])
    def revoke_key(key_id: str, user: User) -> dict:
        if not get_engine().auth.revoke_api_key(user, key_id):
            raise HTTPException(404, "key not found")
        get_engine().audit.record(user.user_id, "apikey.revoke", key_id)
        return {"revoked": key_id}

    # ---------------------------------------------------------------- users (admin)
    @app.post("/users", tags=["admin"], status_code=201)
    def create_user(req: CreateUserRequest, user: User) -> dict:
        require(user, "manage_users")
        try:
            p = get_engine().auth.create_user(req.email, req.password, req.name, req.role,
                                              req.groups)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        get_engine().audit.record(user.user_id, "user.create", p.user_id, role=p.role)
        return _principal_dict(p)

    @app.get("/users", tags=["admin"])
    def list_users(user: User) -> list[dict]:
        require(user, "manage_users")
        return [_principal_dict(p) for p in get_engine().auth.list_users()]

    @app.patch("/users/{user_id}", tags=["admin"])
    def update_user(user_id: str, req: UpdateUserRequest, user: User) -> dict:
        require(user, "manage_users")
        e = get_engine()
        try:
            e.auth.get_user(user_id)
        except AuthError as ex:
            raise HTTPException(404, "user not found") from ex
        e.auth.update_user(user_id, req.role, req.groups, req.disabled)
        e.audit.record(user.user_id, "user.update", user_id, **req.model_dump(exclude_none=True))
        return {"updated": user_id}

    # ---------------------------------------------------------------- documents
    @app.post("/documents", tags=["documents"], status_code=201)
    async def upload(user: User, file: Annotated[UploadFile, File()],
                     groups: Annotated[str, Form()] = "everyone",
                     tags: Annotated[str, Form()] = "") -> dict:
        require(user, "upload_docs")
        e = get_engine()
        group_list = [g.strip().lower() for g in groups.split(",") if g.strip()] or ["everyone"]
        # You can only share a document with groups you belong to (admins: any group).
        if not user.is_admin and not set(group_list) <= user.groups:
            raise PermissionDenied("you can only share documents with groups you belong to: "
                                   + ", ".join(sorted(user.groups)))
        data = await file.read(e.store.max_bytes + 1)
        try:
            doc = e.store.add(file.filename or "upload.txt", data, group_list,
                              [t for t in tags.split(",") if t.strip()], owner=user)
        except DocumentTooLarge as ex:
            raise HTTPException(413, str(ex)) from ex
        except UnsupportedFile as ex:
            raise HTTPException(415, str(ex)) from ex
        except ValueError as ex:
            raise HTTPException(400, str(ex)) from ex
        e.audit.record(user.user_id, "document.upload", doc.id, filename=doc.filename,
                       groups=doc.groups, duplicate=doc.duplicate, chunks=doc.n_chunks)
        return doc.to_dict()

    @app.get("/documents", tags=["documents"])
    def list_documents(user: User, tag: str | None = None) -> list[dict]:
        require(user, "read_docs")
        return [d.to_dict() for d in get_engine().store.list(user, tag)]

    @app.get("/documents/{doc_id}", tags=["documents"])
    def get_document(doc_id: str, user: User) -> dict:
        doc = get_engine().store.get(doc_id)
        # 404 (not 403) for documents you can't read, so their existence isn't revealed.
        if not doc or not user.can_read(doc.groups):
            raise HTTPException(404, "document not found")
        return doc.to_dict()

    @app.delete("/documents/{doc_id}", tags=["documents"])
    def delete_document(doc_id: str, user: User) -> dict:
        e = get_engine()
        doc = e.store.get(doc_id)
        if not doc or not user.can_read(doc.groups):
            raise HTTPException(404, "document not found")
        if doc.owner_id != user.user_id and not user.can("manage_all_docs"):
            raise PermissionDenied("only the owner or an admin can delete this document")
        e.store.delete(doc_id)
        e.audit.record(user.user_id, "document.delete", doc_id, title=doc.title)
        return {"deleted": doc_id}

    # ---------------------------------------------------------------- ask + agent
    @app.post("/ask", tags=["assistant"])
    def ask(req: AskRequest, user: User) -> dict:
        require(user, "ask")
        e = get_engine()
        ans = e.ask(req.question, user, req.k, req.mode, req.tags, req.doc_ids)
        e.audit.record(user.user_id, "ask", None, question=req.question[:200],
                       cited=[c.doc_id for c in ans.citations],
                       trace_id=ans.trace.get("trace_id"))
        return asdict(ans)

    @app.post("/agent", tags=["assistant"])
    def run_agent(req: AgentRequest, user: User) -> dict:
        require(user, "agent")
        return asdict(Agent(get_engine(), user).run(req.task))

    @app.get("/tools", tags=["assistant"])
    def list_tools(user: User) -> list[dict]:
        return [{"name": t.spec.name, "description": t.spec.description,
                 "integration": t.integration, "requires_approval": t.requires_approval}
                for t in build_tools(get_engine(), user).values()]

    # ---------------------------------------------------------------- approvals
    @app.get("/actions", tags=["approvals"])
    def list_actions(user: User, status: str | None = "pending") -> list[dict]:
        return ApprovalQueue(get_engine()).list(user, status)

    @app.post("/actions/{action_id}/decision", tags=["approvals"])
    def decide(action_id: str, req: DecisionRequest, user: User) -> dict:
        q = ApprovalQueue(get_engine())
        action = q.get(action_id)
        if not action or (action["user_id"] != user.user_id and not user.can("approve_any")):
            raise HTTPException(404, "action not found")
        try:
            return q.decide(action_id, user, req.approve)
        except ValueError as ex:
            raise HTTPException(409, str(ex)) from ex

    # ---------------------------------------------------------------- memory + audit
    @app.get("/me/memory", tags=["personalization"])
    def get_memory(user: User) -> dict:
        return asdict(get_engine().memory.get(user.user_id))

    @app.delete("/me/memory", tags=["personalization"])
    def delete_memory(user: User) -> dict:
        get_engine().memory.forget(user.user_id)
        return {"deleted": user.user_id}

    @app.get("/audit", tags=["admin"])
    def audit(user: User, limit: int = 100, user_id: str | None = None,
              action: str | None = None) -> list[dict]:
        require(user, "view_audit")
        return get_engine().audit.list(limit, user_id, action)

    return app


def _principal_dict(p: Principal) -> dict:
    return {"id": p.user_id, "email": p.email, "name": p.name, "role": p.role,
            "groups": sorted(p.groups)}


app = create_app()
