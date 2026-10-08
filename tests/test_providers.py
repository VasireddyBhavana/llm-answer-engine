"""Check the real provider clients' wire format without network access, via httpx.MockTransport."""

import json

import httpx

from grounded.llm import AnthropicLLM, OpenAILLM, ToolCall, ToolSpec

TOOLS = [ToolSpec("calculator", "math", {"type": "object", "properties": {}})]
HISTORY = [
    {"role": "user", "content": "what is 2+2?"},
    {"role": "assistant", "content": "", "tool_calls": [ToolCall("t1", "calculator",
                                                                 {"expression": "2+2"})]},
    {"role": "tool", "tool_call_id": "t1", "name": "calculator", "content": "4"},
]


def fake(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_request_and_tool_use_parsing():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        seen["headers"] = dict(req.headers)
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "Let me check."},
                        {"type": "tool_use", "id": "t2", "name": "calculator",
                         "input": {"expression": "4*2"}}],
            "usage": {"input_tokens": 11, "output_tokens": 7}})

    llm = AnthropicLLM(model="m", api_key="k")
    llm.http = fake(handler)
    resp = llm.complete("sys", HISTORY, tools=TOOLS)

    assert seen["system"] == "sys" and seen["tools"][0]["input_schema"]
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["messages"][1]["content"][0] == {"type": "tool_use", "id": "t1",
                                                  "name": "calculator",
                                                  "input": {"expression": "2+2"}}
    assert seen["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "4"}]}
    assert resp.text == "Let me check." and resp.tool_calls[0].arguments == {"expression": "4*2"}
    assert (resp.input_tokens, resp.output_tokens) == (11, 7)


def test_openai_request_and_tool_call_parsing():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        seen["url"] = str(req.url)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c9", "type": "function",
                 "function": {"name": "calculator", "arguments": "{\"expression\": \"1+1\"}"}}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}})

    llm = OpenAILLM(model="m", base_url="http://localhost:11434/v1", api_key="k")
    llm.http = fake(handler)
    resp = llm.complete("sys", HISTORY, tools=TOOLS)

    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"expression": "2+2"}'
    assert seen["messages"][3] == {"role": "tool", "tool_call_id": "t1", "content": "4"}
    assert resp.tool_calls[0].name == "calculator" and resp.text == ""
