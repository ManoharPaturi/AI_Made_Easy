"""A real MCP client that talks to ``python -m ai_made_easy.mcp.server`` over stdio.

Runs in its own process (``test_mcp_stdio_round_trip``): on Windows the SDK's stdio
client crashed the pytest worker that hosted it. Prints one JSON line with the results;
the test checks them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "ai_made_easy" / "samples" / "mlp_mnist.json"


async def scenario() -> dict:
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "ai_made_easy.mcp.server"], cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            graph = json.loads(SAMPLE.read_text(encoding="utf-8"))
            archs = await session.call_tool("list_blocks", {"category": "Architectures"})
            valid = await session.call_tool("validate_graph", {"graph": graph})
            summary = await session.call_tool("summarize_model", {"graph": graph})
            return {"tools": sorted(t.name for t in tools.tools),
                    "architectures": json.loads(archs.content[0].text)["count"],
                    "valid": json.loads(valid.content[0].text)["valid"],
                    "params": json.loads(summary.content[0].text)["total_params"]}


if __name__ == "__main__":
    print("RESULT " + json.dumps(anyio.run(scenario)), flush=True)
