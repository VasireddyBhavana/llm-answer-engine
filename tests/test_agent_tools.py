import json

import httpx
import pytest

from grounded.agent import Agent, ApprovalQueue, build_tools, safe_eval
from grounded.auth import PermissionDenied
from grounded.integrations import IntegrationError, SafeHTTP, build_integration_tools
from grounded.llm import LLMResponse, ToolCall
from grounded.rag import parse_citations

# ------------------------------------------------------------------ RAG basics


def test_ask_returns_valid_citations(engine):
    ans = engine.ask("When is BM25 better than dense embeddings?")
    assert ans.citations and all(1 <= c.n <= len(ans.contexts) for c in ans.citations)
    assert any(c.doc_id == "embeddings" for c in ans.citations)
    assert [s["name"] for s in ans.trace["spans"]] == ["retrieve", "generate",
                                                       "validate_citations"]


def test_hallucinated_citations_are_stripped(engine):
    hits = engine.retrieve("chunk size", k=2)
    text, cites = parse_citations("Claim one [1]. Made-up claim [7].", hits)
    assert "[7]" not in text and [c.n for c in cites] == [1]


def test_traces_are_written(engine, settings):
    engine.ask("What is HNSW used for?")
    line = json.loads(settings.trace_file.read_text().splitlines()[-1])
    assert line["kind"] == "rag" and line["input"] == "What is HNSW used for?"


# ------------------------------------------------------------------ tools + roles


def test_tools_are_filtered_by_role(engine, users):
    viewer_tools = set(build_tools(engine, users["viewer"]))
    member_tools = set(build_tools(engine, users["hr"]))
    assert "get_weather" not in viewer_tools and "post_message" not in viewer_tools
    assert {"get_weather", "search_wikipedia", "post_message"} <= member_tools


def test_viewer_cannot_run_agent(engine, users):
    with pytest.raises(PermissionDenied):
        Agent(engine, users["viewer"]).run("What is 2+2?")


def test_agent_uses_calculator_and_search(engine, users):
    res = Agent(engine, users["hr"]).run("What is reciprocal rank fusion and what is 1/(60+1)?")
    assert [s.tool for s in res.steps] == ["calculator", "search_knowledge_base"]
    assert "0.016393" in res.answer and res.stopped_reason == "final_answer"


def test_agent_calls_external_apis(engine, users):
    res = Agent(engine, users["eng"]).run("What's the weather in Tokyo and what is 21*2?")
    assert [s.tool for s in res.steps] == ["calculator", "get_weather"]
    assert "Tokyo" in res.answer and "42" in res.answer


def test_agent_search_respects_caller_access(engine, users):
    engine.store.add("bands.md", b"# Bands\n\nThe L5 salary band is secret CANARY-9.",
                     groups=["hr"])
    res = Agent(engine, users["eng"]).run("According to our docs, what is the L5 salary band?")
    assert all("CANARY-9" not in s.output for s in res.steps)


def test_side_effects_wait_for_approval_then_execute_once(engine, users):
    sent = []
    engine.integration_tools = build_integration_tools(
        ["webhook"], "https://hooks.test.local/x",
        transport=httpx.MockTransport(lambda r: sent.append(json.loads(r.content))
                                      or httpx.Response(200)))
    res = Agent(engine, users["hr"]).run("Post to the team channel: offsite moved to Friday")
    assert res.steps[0].status == "pending_approval" and not sent
    q = ApprovalQueue(engine)
    aid = res.pending_actions[0]
    with pytest.raises(PermissionDenied):
        q.decide(aid, users["eng"], True)  # another member can't approve it
    assert q.decide(aid, users["hr"], True)["status"] == "approved"
    assert sent == [{"text": "offsite moved to Friday"}]
    with pytest.raises(ValueError):
        q.decide(aid, users["admin"], True)  # can't execute twice


def test_rejected_action_never_runs(engine, users):
    res = Agent(engine, users["hr"]).run("Send a message to the team channel: hello")
    q = ApprovalQueue(engine)
    assert q.decide(res.pending_actions[0], users["admin"], False)["status"] == "rejected"
    assert q.list(users["hr"]) == []


def test_tool_calls_are_audited(engine, users):
    Agent(engine, users["hr"]).run("What's the weather in Oslo?")
    actions = [(r["action"], r["resource"]) for r in engine.audit.list(10)]
    assert ("tool.call", "get_weather") in actions and ("agent.run", None) in actions


def test_agent_stops_at_max_steps(engine, users):
    class Looping:
        name = "looping"

        def complete(self, system, messages, tools=None, **kw):
            return LLMResponse("", [ToolCall(f"c{len(messages)}", "calculator",
                                             {"expression": "1+1"})])
    res = Agent(engine, users["hr"], llm=Looping(), max_steps=3).run("loop forever")
    assert res.stopped_reason == "max_steps" and len(res.steps) == 3


def test_unknown_or_forbidden_tool_calls_are_denied(engine, users):
    class Sneaky:
        name = "sneaky"

        def complete(self, system, messages, tools=None, **kw):
            if any(m["role"] == "tool" for m in messages):
                return LLMResponse("done")
            return LLMResponse("", [ToolCall("c1", "delete_everything", {})])
    res = Agent(engine, users["hr"], llm=Sneaky()).run("x")
    assert res.steps[0].status == "denied"


def test_tool_errors_returned_not_raised(engine, users):
    agent = Agent(engine, users["hr"])
    assert agent._run_tool("calculator", {"expression": "__import__('os')"}).status == "error"


@pytest.mark.parametrize("expr,expected", [("2+3*4", 14), ("(1+1)**3", 8), ("-5 % 3", 1)])
def test_safe_eval(expr, expected):
    assert safe_eval(expr) == expected


def test_safe_eval_rejects_code():
    with pytest.raises(ValueError):
        safe_eval("open('x')")


# ------------------------------------------------------------------ SafeHTTP


def test_safe_http_blocks_unlisted_hosts_and_http():
    http = SafeHTTP({"api.example.com"}, transport=httpx.MockTransport(lambda r:
                                                                       httpx.Response(200)))
    with pytest.raises(IntegrationError, match="not an allowed"):
        http.request("GET", "https://evil.example.net/steal")
    with pytest.raises(IntegrationError):
        http.request("GET", "http://api.example.com/plain-http")
    with pytest.raises(IntegrationError):
        http.request("GET", "https://169.254.169.254/latest/meta-data")  # cloud metadata SSRF


def test_safe_http_retries_then_succeeds():
    calls, sleeps = [], []

    def handler(req):
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json={"ok": 1})
    http = SafeHTTP({"api.example.com"}, retries=2, transport=httpx.MockTransport(handler),
                    sleep=sleeps.append)
    assert http.get_json("https://api.example.com/x") == {"ok": 1}
    assert len(calls) == 3 and sleeps == [0.3, 0.6]


def test_safe_http_gives_up_and_caps_size():
    http = SafeHTTP({"a.example.com"}, retries=1, sleep=lambda s: None,
                    transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(IntegrationError, match="after 2 attempts"):
        http.request("GET", "https://a.example.com/")
    big = SafeHTTP({"a.example.com"}, max_bytes=10,
                   transport=httpx.MockTransport(lambda r: httpx.Response(200, text="x" * 50)))
    with pytest.raises(IntegrationError, match="too large"):
        big.request("GET", "https://a.example.com/")


def test_weather_tool_formats_open_meteo(engine, users):
    out = build_tools(engine, users["hr"])["get_weather"].fn(city="lisbon")
    assert out.startswith("Weather in Lisbon") and "18.0°C" in out and "open-meteo" in out
