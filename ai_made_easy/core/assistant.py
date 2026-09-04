"""Assistant client: pure-Python chat against an OpenAI-compatible endpoint.

Configured via environment variables (no keys in code or project files):
  AIME_ASSISTANT_BASE_URL  e.g. https://api.openai.com/v1
  AIME_ASSISTANT_API_KEY   the API key
  AIME_ASSISTANT_MODEL     e.g. gpt-4o-mini

The system prompt embeds the block catalog + the current graph so the model
can answer questions and propose complete corrected graph JSON, which the
UI validates before applying.
"""
from __future__ import annotations

import json
import os
import urllib.request


class AssistantError(RuntimeError):
    pass


def assistant_config() -> dict:
    return {
        "base_url": os.environ.get("AIME_ASSISTANT_BASE_URL", "").rstrip("/"),
        "api_key": os.environ.get("AIME_ASSISTANT_API_KEY", ""),
        "model": os.environ.get("AIME_ASSISTANT_MODEL", ""),
    }


def is_configured() -> bool:
    cfg = assistant_config()
    return bool(cfg["base_url"] and cfg["api_key"] and cfg["model"])


def build_system_prompt(graph_json: dict, block_catalog: list[dict]) -> str:
    catalog = [
        {"type_id": b["type_id"], "category": b["category"],
         "params": [p["name"] for p in b["params"]]}
        for b in block_catalog
    ]
    return (
        "You are the AI assistant inside 'AI Made Easy', a visual block-based "
        "model builder. The user designs neural networks as graphs of blocks.\n\n"
        "BLOCK CATALOG (type_id / category / param names):\n"
        + json.dumps(catalog, indent=0)
        + "\n\nCURRENT GRAPH JSON:\n"
        + json.dumps(graph_json, indent=1)
        + "\n\nRULES:\n"
        "- Answer questions about the graph, ML design, or the app.\n"
        "- When the user asks for changes, output the COMPLETE corrected "
        "graph JSON in one ```json fenced block (same schema as CURRENT "
        "GRAPH), using only blocks from the catalog.\n"
        "- Shapes are channels-first without batch dim; Dense needs flat "
        "input (add core.flatten); every non-Input block needs its inputs "
        "connected; exactly one core.input and one core.output.\n"
        "- Keep node ids stable when editing.\n"
    )


def chat(messages: list[dict], timeout: float = 120.0) -> str:
    cfg = assistant_config()
    if not is_configured():
        raise AssistantError(
            "set AIME_ASSISTANT_BASE_URL, AIME_ASSISTANT_API_KEY and "
            "AIME_ASSISTANT_MODEL to enable the assistant"
        )
    payload = json.dumps({"model": cfg["model"], "messages": messages}).encode()
    request = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except Exception as exc:
        raise AssistantError(f"assistant request failed: {exc}") from exc
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise AssistantError(f"unexpected assistant reply: {body}") from exc


def extract_graph(reply: str) -> dict | None:
    """Pull the first ```json fenced block out of a reply, if any."""
    if "```json" not in reply:
        return None
    chunk = reply.split("```json", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None
