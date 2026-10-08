import json

import pytest

from grounded.agent import Agent, safe_eval
from grounded.llm import LLMResponse, ToolCall
from grounded.rag import parse_citations


def test_ask_returns_valid_citations(engine):
    ans = engine.ask("When is BM25 better than dense embeddings?")
    assert ans.citations, ans.text
    assert all(1 <= c.n <= len(ans.contexts) for c in ans.citations)
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


def test_agent_uses_calculator_and_search(engine):
    res = Agent(engine).run("What is reciprocal rank fusion and what is 1/(60+1)?")
    tools = [s.tool for s in res.steps]
    assert tools == ["calculator", "search_knowledge_base"]
    assert "0.016393" in res.answer
    assert res.stopped_reason == "final_answer"


def test_agent_stops_at_max_steps(engine):
    class Looping:
        name = "looping"
        def complete(self, system, messages, tools=None, **kw):
            return LLMResponse("", [ToolCall(f"c{len(messages)}", "calculator",
                                             {"expression": "1+1"})])
    res = Agent(engine, llm=Looping(), max_steps=3).run("loop forever")
    assert res.stopped_reason == "max_steps" and len(res.steps) == 3


def test_agent_reports_tool_errors_instead_of_crashing(engine):
    agent = Agent(engine)
    assert agent._run_tool("calculator", {"expression": "__import__('os')"}).startswith("Error")
    assert agent._run_tool("nope", {}).startswith("Error: unknown tool")


@pytest.mark.parametrize("expr,expected", [("2+3*4", 14), ("(1+1)**3", 8), ("-5 % 3", 1)])
def test_safe_eval(expr, expected):
    assert safe_eval(expr) == expected


def test_safe_eval_rejects_code():
    with pytest.raises(ValueError):
        safe_eval("open('x')")
