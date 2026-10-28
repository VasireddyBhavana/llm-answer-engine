"""External API integrations, exposed to the agent as tools.

Every outbound call goes through `SafeHTTP`, which enforces what production integrations need:
  * host allowlist      — the agent can only reach hosts an integration declared
  * timeouts            — a slow API cannot hang a request
  * retries w/ backoff  — on 429 / 5xx / connection errors, honouring Retry-After
  * response size cap   — a huge response cannot blow up memory or the prompt
  * no secrets in logs  — callers pass headers; nothing is echoed back to the model

Integrations shipped (no API keys needed for the first two):
  weather    Open-Meteo geocoding + forecast           read-only
  wikipedia  Wikipedia search + page summary            read-only
  webhook    Slack-compatible incoming webhook          SIDE EFFECT -> needs human approval
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from .llm import ToolSpec


class IntegrationError(Exception):
    pass


class SafeHTTP:
    def __init__(self, allowed_hosts: set[str], timeout_s: float = 8.0, retries: int = 2,
                 max_bytes: int = 1_000_000, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.allowed = {h.lower() for h in allowed_hosts}
        self.retries, self.max_bytes, self.sleep = retries, max_bytes, sleep
        self.client = httpx.Client(timeout=timeout_s, transport=transport,
                                   follow_redirects=False,
                                   headers={"User-Agent": "llm-answer-engine/0.2"})

    def request(self, method: str, url: str, **kw: Any) -> httpx.Response:
        host = (urlparse(url).hostname or "").lower()
        if urlparse(url).scheme != "https" or host not in self.allowed:
            raise IntegrationError(f"blocked: {host or url} is not an allowed https host")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self.client.request(method, url, **kw)
            except httpx.TransportError as e:  # DNS, connect, read timeout...
                last = e
            else:
                if r.status_code not in (429, 500, 502, 503, 504):
                    if len(r.content) > self.max_bytes:
                        raise IntegrationError("response too large")
                    if r.status_code >= 400:
                        raise IntegrationError(f"{host} returned HTTP {r.status_code}")
                    return r
                last = IntegrationError(f"{host} returned HTTP {r.status_code}")
                retry_after = r.headers.get("retry-after", "")
                if retry_after.isdigit() and attempt < self.retries:
                    self.sleep(min(float(retry_after), 5.0))
                    continue
            if attempt < self.retries:
                self.sleep(0.3 * 2 ** attempt)  # exponential backoff
        raise IntegrationError(f"request to {host} failed after {self.retries + 1} attempts: "
                               f"{last}")

    def get_json(self, url: str, **kw: Any) -> Any:
        return self.request("GET", url, **kw).json()


@dataclass
class Tool:
    """A capability the agent may call.

    min_permission     who may use it (checked against the caller's role)
    requires_approval  side effects: the call is queued for a human instead of executed
    """

    spec: ToolSpec
    fn: Callable[..., str]
    min_permission: str = "agent"
    requires_approval: bool = False
    integration: str = "builtin"
    tags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- integrations

WMO_CODES = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
             48: "rime fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
             61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
             75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "violent showers",
             95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def weather_tools(http: SafeHTTP) -> list[Tool]:
    def get_weather(city: str) -> str:
        geo = http.get_json("https://geocoding-api.open-meteo.com/v1/search",
                            params={"name": city, "count": 1})
        if not geo.get("results"):
            return f"No location found for '{city}'."
        place = geo["results"][0]
        fc = http.get_json("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min", "forecast_days": 1,
            "timezone": "auto"})
        cur, daily = fc["current"], fc.get("daily", {})
        desc = WMO_CODES.get(cur.get("weather_code"), "unknown conditions")
        hi = (daily.get("temperature_2m_max") or ["?"])[0]
        lo = (daily.get("temperature_2m_min") or ["?"])[0]
        return (f"Weather in {place['name']}, {place.get('country', '')}: {desc}, "
                f"{cur['temperature_2m']}°C now (high {hi}°C, low {lo}°C), "
                f"wind {cur['wind_speed_10m']} km/h. Source: open-meteo.com")

    return [Tool(ToolSpec(
        "get_weather", "Get current weather and today's high/low for a city.",
        {"type": "object", "properties": {"city": {"type": "string",
                                                    "description": "City name, e.g. 'Seattle'"}},
         "required": ["city"]}), get_weather, "use_integrations", integration="weather")]


def wikipedia_tools(http: SafeHTTP) -> list[Tool]:
    def search_wikipedia(query: str) -> str:
        res = http.get_json("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": 3,
            "format": "json"})
        hits = res.get("query", {}).get("search", [])
        if not hits:
            return f"No Wikipedia results for '{query}'."
        title = hits[0]["title"]
        page = http.get_json("https://en.wikipedia.org/api/rest_v1/page/summary/" +
                             quote(title.replace(" ", "_"), safe=""))
        others = ", ".join(h["title"] for h in hits[1:])
        return (f"{page.get('title', title)}: {page.get('extract', '').strip()} "
                f"(Source: https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))})"
                + (f"\nRelated: {others}" if others else ""))

    return [Tool(ToolSpec(
        "search_wikipedia", "Look up general public knowledge on Wikipedia. Returns the summary "
        "of the best-matching article with its URL. Use for facts not in the company knowledge "
        "base.", {"type": "object", "properties": {"query": {"type": "string"}},
                  "required": ["query"]}), search_wikipedia, "use_integrations",
        integration="wikipedia")]


def webhook_tools(http: SafeHTTP, webhook_url: str) -> list[Tool]:
    def post_message(text: str) -> str:
        if not webhook_url:
            raise IntegrationError("no webhook configured (set GROUNDED_WEBHOOK_URL)")
        http.request("POST", webhook_url, json={"text": text[:3000]})
        return "Message posted to the team channel."

    return [Tool(ToolSpec(
        "post_message", "Post a message to the team's chat channel. This sends a real message, "
        "so it is queued for the user's approval before it is sent.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
        post_message, "use_integrations", requires_approval=True, integration="webhook")]


def build_integration_tools(names: tuple[str, ...] | list[str], webhook_url: str = "",
                            timeout_s: float = 8.0,
                            transport: httpx.BaseTransport | None = None) -> list[Tool]:
    hosts = {"geocoding-api.open-meteo.com", "api.open-meteo.com", "en.wikipedia.org"}
    if webhook_url:
        hosts.add((urlparse(webhook_url).hostname or "").lower())
    http = SafeHTTP(hosts, timeout_s=timeout_s, transport=transport)
    tools: list[Tool] = []
    if "weather" in names:
        tools += weather_tools(http)
    if "wikipedia" in names:
        tools += wikipedia_tools(http)
    if "webhook" in names:
        tools += webhook_tools(http, webhook_url)
    return tools
