"""A real MCP client that talks to ``python -m ai_made_easy.mcp.server`` over stdio.

Runs in its own process (``test_mcp_stdio_round_trip``): on Windows the SDK's stdio
client crashed the pytest worker that hosted it. Prints a ``STEP`` line as each step
finishes and one ``RESULT {json}`` line as soon as the calls are done, then exits without
waiting for the SDK to shut the server down (the test reads up to RESULT). Each step has
its own time limit, so a hang names the step; the server dumps its threads' stacks if
it is still running after WATCHDOG seconds. The server's stderr goes to argv[1].
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "ai_made_easy" / "samples" / "mlp_mnist.json"
STEP_SECONDS = 120
WATCHDOG = 90
START = time.monotonic()


def say(line: str) -> None:
    print(f"{line} ({time.monotonic() - START:.1f}s)", flush=True)


async def step(name: str, call):  # noqa: ANN001, ANN201
    try:
        with anyio.fail_after(STEP_SECONDS):
            out = await call
    except TimeoutError:
        say(f"STUCK {name}")
        os._exit(3)
    say(f"STEP {name}")
    return out


async def scenario(errlog) -> None:  # noqa: ANN001
    # the server dumps every thread's stack to its stderr if it is still running after
    # WATCHDOG seconds (before a step's limit), so a hang shows where it is
    watchdog = ("import faulthandler, runpy; "
                f"faulthandler.dump_traceback_later({WATCHDOG}, exit=True); "
                "runpy.run_module('ai_made_easy.mcp.server', run_name='__main__')")
    params = StdioServerParameters(command=sys.executable, args=["-c", watchdog],
                                   cwd=str(ROOT))
    say("STEP start")
    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            await step("initialize", session.initialize())
            tools = await step("list_tools", session.list_tools())
            graph = json.loads(SAMPLE.read_text(encoding="utf-8"))
            archs = await step("list_blocks",
                               session.call_tool("list_blocks", {"category": "Architectures"}))
            valid = await step("validate_graph",
                               session.call_tool("validate_graph", {"graph": graph}))
            summary = await step("summarize_model",
                                 session.call_tool("summarize_model", {"graph": graph}))
            print("RESULT " + json.dumps({
                "tools": sorted(t.name for t in tools.tools),
                "architectures": json.loads(archs.content[0].text)["count"],
                "valid": json.loads(valid.content[0].text)["valid"],
                "params": json.loads(summary.content[0].text)["total_params"]}), flush=True)
            os._exit(0)          # the test has what it needs; don't wait on the shutdown


if __name__ == "__main__":
    with open(sys.argv[1], "w", encoding="utf-8") as errlog:
        anyio.run(scenario, errlog)
