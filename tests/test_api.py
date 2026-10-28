from conftest import login
from test_loaders import make_pdf

from grounded.api import create_app

SECRET = b"# Bands\n\n## L5\n\nThe L5 salary band is 185k to 230k. Code CANARY-API-1."


def upload(client, headers, name, data, groups="everyone", tags=""):
    return client.post("/documents", headers=headers, files={"file": (name, data)},
                       data={"groups": groups, "tags": tags})


def test_health_is_public_everything_else_needs_auth(client):
    assert client.get("/health").json()["status"] == "ok"
    for method, path in [("post", "/ask"), ("get", "/documents"), ("get", "/auth/me"),
                         ("post", "/agent"), ("get", "/audit")]:
        body = {"question": "x", "task": "x"}
        r = client.post(path, json=body) if method == "post" else client.get(path)
        assert r.status_code == 401, path
    assert client.get("/documents", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_login_me_and_bad_password(client, users):
    h = login(client, "hr@corp.test", "hr-password-1")
    me = client.get("/auth/me", headers=h).json()
    assert me["role"] == "member" and me["groups"] == ["everyone", "hr"]
    r = client.post("/auth/login", json={"email": "hr@corp.test", "password": "wrong-password"})
    assert r.status_code == 401


def test_api_key_flow(client, headers):
    r = client.post("/auth/api-keys", headers=headers["eng"], json={"name": "ci-bot"})
    key = r.json()["api_key"]
    kh = {"Authorization": f"Bearer {key}"}
    assert client.get("/auth/me", headers=kh).json()["email"] == "eng@corp.test"
    kid = r.json()["id"]
    assert client.delete(f"/auth/api-keys/{kid}", headers=headers["eng"]).status_code == 200
    assert client.get("/auth/me", headers=kh).status_code == 401


def test_admin_only_endpoints(client, headers):
    assert client.get("/users", headers=headers["hr"]).status_code == 403
    assert client.get("/audit", headers=headers["hr"]).status_code == 403
    r = client.post("/users", headers=headers["admin"], json={
        "email": "new@corp.test", "password": "new-password", "role": "viewer",
        "groups": ["finance"]})
    assert r.status_code == 201 and r.json()["groups"] == ["everyone", "finance"]
    uid = r.json()["id"]
    assert client.patch(f"/users/{uid}", headers=headers["admin"],
                        json={"role": "member"}).status_code == 200
    assert client.patch("/users/u_missing", headers=headers["admin"],
                        json={"role": "member"}).status_code == 404
    assert client.post("/users", headers=headers["admin"], json={
        "email": "new@corp.test", "password": "new-password"}).status_code == 400


def test_upload_pdf_then_ask_with_citation(client, headers):
    r = upload(client, headers["hr"], "leave.pdf",
               make_pdf(["Parental leave is 16 weeks fully paid."]), groups="hr", tags="policy")
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["mime"] == "application/pdf" and doc["groups"] == ["hr"]
    ans = client.post("/ask", headers=headers["hr"],
                      json={"question": "How long is parental leave?"}).json()
    assert "16 weeks" in ans["text"]
    assert ans["citations"][0]["doc_id"] == doc["id"]
    assert ans["citations"][0]["section"] == "Page 1"


def test_cross_group_isolation_over_http(client, headers):
    doc = upload(client, headers["hr"], "bands.md", SECRET, groups="hr").json()
    for who in ("eng", "viewer"):
        ans = client.post("/ask", headers=headers[who],
                          json={"question": "What is the L5 salary band?"}).json()
        assert "CANARY-API-1" not in str(ans)
        ids = [d["id"] for d in client.get("/documents", headers=headers[who]).json()]
        assert doc["id"] not in ids
        # 404, not 403: don't reveal that the document exists
        assert client.get(f"/documents/{doc['id']}", headers=headers[who]).status_code == 404
        assert client.delete(f"/documents/{doc['id']}",
                             headers=headers[who]).status_code == 404


def test_cannot_share_into_groups_you_are_not_in(client, headers):
    assert upload(client, headers["eng"], "x.md", b"# X\n\nhi", groups="hr").status_code == 403
    assert upload(client, headers["viewer"], "x.md", b"# X\n\nhi").status_code == 403
    assert upload(client, headers["admin"], "x.md", b"# X\n\nhi", groups="hr").status_code == 201


def test_upload_validation(client, headers):
    assert upload(client, headers["hr"], "a.exe", b"MZ").status_code == 415
    assert upload(client, headers["hr"], "e.md", b"  ").status_code == 415


def test_delete_owner_or_admin_only(client, headers, users, engine):
    doc = upload(client, headers["admin"], "pub.md", b"# Pub\n\npublic note").json()
    assert client.delete(f"/documents/{doc['id']}", headers=headers["hr"]).status_code == 403
    assert client.delete(f"/documents/{doc['id']}", headers=headers["admin"]).status_code == 200


def test_ask_validation_and_modes(client, headers):
    assert client.post("/ask", headers=headers["viewer"], json={"question": ""}).status_code == 422
    assert client.post("/ask", headers=headers["viewer"],
                       json={"question": "x", "mode": "magic"}).status_code == 422
    r = client.post("/ask", headers=headers["viewer"],
                    json={"question": "What is nucleus sampling?", "mode": "hybrid+rerank"})
    assert r.status_code == 200 and r.json()["citations"]


def test_agent_tools_and_approvals_over_http(client, headers):
    assert client.post("/agent", headers=headers["viewer"],
                       json={"task": "What is 6*7?"}).status_code == 403
    names = {t["name"] for t in client.get("/tools", headers=headers["hr"]).json()}
    assert {"get_weather", "post_message", "search_knowledge_base"} <= names

    res = client.post("/agent", headers=headers["hr"],
                      json={"task": "Post to the team channel: release shipped"}).json()
    aid = res["pending_actions"][0]
    assert [a["id"] for a in client.get("/actions", headers=headers["hr"]).json()] == [aid]
    assert client.get("/actions", headers=headers["eng"]).json() == []
    assert client.post(f"/actions/{aid}/decision", headers=headers["eng"],
                       json={"approve": True}).status_code == 404
    r = client.post(f"/actions/{aid}/decision", headers=headers["hr"], json={"approve": True})
    assert r.json()["status"] == "approved"
    assert client.post(f"/actions/{aid}/decision", headers=headers["hr"],
                       json={"approve": True}).status_code == 409


def test_audit_log_records_security_events(client, headers):
    client.post("/auth/login", json={"email": "hr@corp.test", "password": "bad-password"})
    upload(client, headers["hr"], "n.md", b"# N\n\nnote", groups="hr")
    client.get("/users", headers=headers["hr"])
    rows = client.get("/audit", headers=headers["admin"]).json()
    seen = {(r["action"], r["outcome"]) for r in rows}
    assert {("auth.login", "denied"), ("document.upload", "ok"),
            ("permission.denied", "denied")} <= seen


def test_memory_endpoints(client, headers):
    client.post("/ask", headers=headers["hr"],
                json={"question": "I'm a beginner, keep it short: what is RAG?"})
    mem = client.get("/me/memory", headers=headers["hr"]).json()
    assert mem["preferences"] == {"level": "beginner", "style": "concise"}
    assert client.delete("/me/memory", headers=headers["hr"]).status_code == 200


def test_rate_limit(engine, users):
    from dataclasses import replace

    from fastapi.testclient import TestClient
    engine.settings = replace(engine.settings, rate_limit_per_min=3)
    c = TestClient(create_app(engine))
    h = login(c, "eng@corp.test", "eng-password-1")
    codes = [c.get("/auth/me", headers=h).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
