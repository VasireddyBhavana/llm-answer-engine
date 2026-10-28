"""LLM clients behind one small interface.

Three backends:
  * MockLLM       — offline and deterministic, so tests/CI/evals run with no API key.
  * AnthropicLLM  — Claude via the Messages API, with native tool use.
  * OpenAILLM     — any OpenAI-compatible Chat Completions server (OpenAI, Ollama, vLLM, Groq...).

Messages use a provider-neutral format:
  {"role": "user" | "assistant", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Each client translates to its provider's wire format.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import Settings
from .text import sentences, tokenize


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- Anthropic


class AnthropicLLM:
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 60):
        self.model = model or "claude-haiku-5-5"
        self.name = f"anthropic:{self.model}"
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.http = httpx.Client(timeout=timeout)

    @staticmethod
    def _to_wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                         "content": m["content"]}
                # Consecutive tool results must share one user turn.
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
            elif m.get("tool_calls"):
                content: list[dict[str, Any]] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                            for c in m["tool_calls"]]
                wire.append({"role": "assistant", "content": content})
            else:
                wire.append({"role": m["role"], "content": m["content"]})
        return wire

    def complete(self, system, messages, tools=None, temperature=0.0, max_tokens=800):
        body: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": self._to_wire(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = [{"name": t.name, "description": t.description,
                              "input_schema": t.parameters} for t in tools]
        r = self.http.post(self.URL, json=body, headers={
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        })
        r.raise_for_status()
        data = r.json()
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        calls = [ToolCall(b["id"], b["name"], b["input"])
                 for b in data["content"] if b["type"] == "tool_use"]
        usage = data.get("usage", {})
        return LLMResponse(text, calls, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


# --------------------------------------------------------------------------- OpenAI-compatible


class OpenAILLM:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 60):
        self.model = model or "gpt-4o-mini"
        self.name = f"openai:{self.model}"
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed-for-local")
        self.http = httpx.Client(timeout=timeout)

    @staticmethod
    def _to_wire(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "tool":
                wire.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                             "content": m["content"]})
            elif m.get("tool_calls"):
                wire.append({"role": "assistant", "content": m.get("content") or None,
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.name,
                                                          "arguments": json.dumps(c.arguments)}}
                                            for c in m["tool_calls"]]})
            else:
                wire.append({"role": m["role"], "content": m["content"]})
        return wire

    def complete(self, system, messages, tools=None, temperature=0.0, max_tokens=800):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_wire(system, messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools]
        r = self.http.post(f"{self.base_url}/chat/completions", json=body,
                           headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = [ToolCall(c["id"], c["function"]["name"],
                          json.loads(c["function"]["arguments"] or "{}"))
                 for c in (msg.get("tool_calls") or [])]
        usage = data.get("usage", {})
        return LLMResponse(msg.get("content") or "", calls,
                           usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


# --------------------------------------------------------------------------- Mock


class MockLLM:
    """A deterministic stand-in for a real model.

    It is NOT intelligent. It does just enough to exercise every code path:
      * RAG prompts  -> extracts context sentences that best overlap the question, with citations.
      * Agent prompts -> routes by keyword to the available tools (calculator, knowledge base,
                         weather, Wikipedia, documents, messaging), then answers from results.
      * Rerank prompts -> scores passages by best-sentence overlap with the query.
      * Judge prompts -> scores by word overlap between answer and context.
    Swap in a real provider with GROUNDED_LLM=anthropic or GROUNDED_LLM=openai.
    """

    name = "mock"

    def complete(self, system, messages, tools=None, temperature=0.0, max_tokens=800):
        last = messages[-1]
        prompt = last["content"] if isinstance(last.get("content"), str) else ""

        if "JUDGE" in system:
            return LLMResponse(self._judge(prompt))

        if "RERANK" in system:
            return LLMResponse(self._rerank(prompt))

        if tools:
            return self._agent_step(messages, tools)

        if "<sources>" in prompt:
            return LLMResponse(self._extractive_answer(prompt))

        return LLMResponse(prompt.strip().splitlines()[-1] if prompt.strip() else "")

    # -- RAG ---------------------------------------------------------------
    @staticmethod
    def _extractive_answer(prompt: str, n: int = 2) -> str:
        question = prompt.split("<question>")[-1].split("</question>")[0]
        q = set(tokenize(question, drop_stopwords=True))
        scored: list[tuple[float, int, str]] = []
        for m in re.finditer(r'<source id="(\d+)"[^>]*>(.*?)</source>', prompt, re.S):
            sid, body = int(m.group(1)), re.sub(r"^\s*\[[^\]]*\]\s*", "", m.group(2))
            for s in sentences(body):
                overlap = len(q & set(tokenize(s, drop_stopwords=True)))
                if overlap:
                    scored.append((overlap / (1 + len(s) / 400), sid, s.strip()))
        if not scored:
            return "I don't know based on the provided sources."
        scored.sort(key=lambda x: -x[0])
        return " ".join(f"{s} [{sid}]" for _, sid, s in scored[:n])

    # -- Agent -------------------------------------------------------------
    @staticmethod
    def _plan(task: str, names: set[str]) -> list[tuple[str, dict[str, Any]]]:
        """Keyword routing that imitates what a real model would decide from tool descriptions."""
        plan: list[tuple[str, dict[str, Any]]] = []
        low = task.lower()
        math = MockLLM._find_arithmetic(task)
        if math and "calculator" in names:
            plan.append(("calculator", {"expression": math}))
        if "get_weather" in names and re.search(r"\b(weather|forecast|temperature)\b", low):
            m = re.search(r"\b(?:in|for|at)\s+([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*)", task)
            plan.append(("get_weather", {"city": m.group(1) if m else "London"}))
        if "search_wikipedia" in names and "wikipedia" in low:
            m = re.search(r"wikipedia\s+(?:for|about|on)?\s*(.+?)(?:[?.!]|\s+and\s|$)", task,
                          re.I)
            plan.append(("search_wikipedia", {"query": (m.group(1) if m else task).strip()}))
        if "list_documents" in names and re.search(
                r"\b(which|what|list)\b.{0,20}\b(documents|docs|files)\b", low):
            plan.append(("list_documents", {}))
        if "post_message" in names and re.search(
                r"\b(post|send|notify|message)\b.{0,40}\b(slack|team|channel|webhook)\b", low):
            text = task.split(":", 1)[1].strip() if ":" in task else task
            plan.append(("post_message", {"text": text}))
        others = [n for n, _ in plan if n != "calculator"]
        only_short_math = plan and not others and len(task) < len(math or "") + 25
        wants_kb = re.search(r"\b(knowledge base|our docs|documents? say|policy|according)\b", low)
        if "search_knowledge_base" in names and ((not plan) or wants_kb
                                                 or (not others and not only_short_math)):
            plan.append(("search_knowledge_base", {"query": task}))
        return plan

    @staticmethod
    def _agent_step(messages, tools) -> LLMResponse:
        names = {t.name for t in tools}
        user_q = next(m["content"] for m in messages if m["role"] == "user")
        tool_results = [m for m in messages if m["role"] == "tool"]
        used = {m["name"] for m in tool_results}
        for name, args in MockLLM._plan(user_q, names):
            if name not in used:
                return LLMResponse("", [ToolCall(f"call_{len(tool_results)}", name, args)])

        parts = []
        for m in tool_results:
            content = re.sub(r"</?tool_output[^>]*>", "", m["content"]).strip()
            if m["name"] == "calculator":
                parts.append(f"The result is {content}.")
            elif m["name"] == "search_knowledge_base":
                parts.append(MockLLM._extractive_answer(
                    f"{content}<question>{user_q}</question>", n=1))
            else:
                parts.append(content.split("\n")[0][:300])
        return LLMResponse(" ".join(parts) or "I could not find an answer.")

    @staticmethod
    def _find_arithmetic(text: str) -> str | None:
        """Longest run of arithmetic characters containing an operator, with balanced parens."""
        runs = [r.strip() for r in re.findall(r"[\d\.\s\+\-\*/\(\)%]+", text)]
        runs = [r for r in runs if re.search(r"\d\s*[\+\-\*/%]\s*[\d(]", r)]
        if not runs:
            return None
        expr = max(runs, key=len)
        while expr.count("(") > expr.count(")") and expr.startswith("("):
            expr = expr[1:].strip()
        while expr.count(")") > expr.count("(") and expr.endswith(")"):
            expr = expr[:-1].strip()
        return expr

    # -- Rerank ------------------------------------------------------------
    @staticmethod
    def _rerank(prompt: str) -> str:
        query = prompt.split("<query>")[-1].split("</query>")[0]
        q = set(tokenize(query, drop_stopwords=True))
        passages = re.findall(r"<passage n=\"\d+\">(.*?)</passage>", prompt, re.S)
        raw = []
        for body in passages:
            body = re.sub(r"^\s*\[[^\]]*\]\s*", "", body)  # ignore the chunk header
            best = max((len(q & set(tokenize(s, drop_stopwords=True))) /
                        (1 + len(s) / 400) for s in sentences(body)), default=0.0)
            raw.append(best)
        top = max(raw, default=0) or 1
        return json.dumps({"scores": [round(10 * r / top, 2) for r in raw]})

    # -- Judge -------------------------------------------------------------
    @staticmethod
    def _judge(prompt: str) -> str:
        ctx = prompt.split("<context>")[-1].split("</context>")[0]
        ans = prompt.split("<answer>")[-1].split("</answer>")[0]
        a = set(tokenize(re.sub(r"\[\d+\]", "", ans), drop_stopwords=True))
        c = set(tokenize(ctx, drop_stopwords=True))
        score = len(a & c) / max(len(a), 1)
        return json.dumps({"score": round(score, 3), "reason": "word-overlap heuristic (mock)"})


def get_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        return AnthropicLLM(model=settings.model or None)
    if provider == "openai":
        return OpenAILLM(model=settings.model or None, base_url=settings.openai_base_url)
    return MockLLM()
