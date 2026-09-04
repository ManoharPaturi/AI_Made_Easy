"""Headless CLI over core — proves the app is drivable without the GUI.

This is the same JSON-in/JSON-out surface a future MCP agent will use.

Usage:
  aime blocks                       # list registered blocks (JSON schemas)
  aime validate <project.json>      # check a graph, print issues
  aime gen <project.json> -f pytorch [-o DIR]   # generate model code
  aime train <project.json> -f pytorch [-o DIR] # generate training script
  aime onnx <project.json> [-o DIR]   # generate (and with --run, execute) an ONNX export script
  aime jit <project.json> [-o DIR]    # generate (and with --run, execute) a TorchScript export script
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from ai_made_easy.core.codegen import FRAMEWORKS, export, export_training, generate
from ai_made_easy.core.codegen import sanitize_identifier as sanitize_name
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aime", description="AI Made Easy")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("blocks", help="list registered blocks")
    p_val = sub.add_parser("validate", help="validate a project graph")
    p_val.add_argument("project", help="path to project .json")
    for name, help_text in (("gen", "generate model code"), ("train", "generate training script")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("project", help="path to project .json")
        p.add_argument("-f", "--framework", choices=FRAMEWORKS, default="pytorch")
        p.add_argument("-o", "--out", default="exports", help="output directory")
    p_run = sub.add_parser("run", help="train headlessly and stream events")
    p_run.add_argument("project", help="path to project .json")
    p_sum = sub.add_parser("summary", help="print the analytic model summary as JSON")
    p_sum.add_argument("project", help="path to project .json")
    p_llm = sub.add_parser("llm", help="generate an LLM workflow script")
    p_llm.add_argument("project", help="path to project .json")
    p_llm.add_argument("-o", "--out", default="exports", help="output directory")
    for name, help_text in (("onnx", "generate ONNX export script"),
                            ("jit", "generate TorchScript export script")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("project", help="path to project .json")
        p.add_argument("-o", "--out", default="exports", help="output directory")
        p.add_argument("--run", action="store_true",
                       help="execute the generated script immediately")

    args = parser.parse_args(argv)

    if args.command == "blocks":
        print(json.dumps(get_registry().list_blocks(), indent=2))
        return 0

    with open(args.project) as fh:
        graph = Graph.from_dict(json.load(fh))

    if args.command == "validate":
        issues = graph.validate()
        for issue in issues:
            print(issue)
        print(f"{len(issues)} issue(s)")
        return 1 if any(i.severity == "error" for i in issues) else 0

    if args.command in ("gen", "train"):
        try:
            path = (
                export(graph, args.framework, args.out)
                if args.command == "gen"
                else export_training(graph, args.framework, args.out)
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(path)
        return 0

    if args.command == "summary":
        from ai_made_easy.core.summary import summarize

        s = summarize(graph)
        print(json.dumps({
            "total_params": s.total_params,
            "layers": [
                {"name": L.name, "type": L.type_id,
                 "output_shape": L.output_shape, "params": L.params}
                for L in s.layers
            ],
        }, indent=2))
        return 0

    if args.command == "run":
        from ai_made_easy.core.runner.manager import RunManager

        mgr = RunManager()
        run_id = mgr.start(graph)
        print(json.dumps({"type": "run_started", "run_id": run_id}), flush=True)
        import time as _time

        seen = 0
        while True:
            st = mgr.status(run_id)
            for event in mgr.get(run_id).events[seen:]:
                print(json.dumps(event), flush=True)
            seen = len(mgr.get(run_id).events)
            if st["state"] in ("finished", "failed", "stopped"):
                print(json.dumps({"type": "run_" + st["state"],
                                  "returncode": st["returncode"]}), flush=True)
                return 0 if st["state"] == "finished" else 1
            _time.sleep(0.1)

    if args.command == "llm":
        from ai_made_easy.core.codegen.llm_gen import generate_llm_script

        try:
            code = generate_llm_script(graph)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        from pathlib import Path

        script = Path(args.out) / f"{sanitize_name(graph.name)}_llm.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(code)
        print(script)
        return 0

    if args.command in ("onnx", "jit"):
        from ai_made_easy.core.codegen.runtime_export import (
            generate_onnx_export,
            generate_torchscript_export,
        )

        try:
            stem = graph.name.replace("-", "_")
            out_model = f"{args.out}/{stem}.{'onnx' if args.command == 'onnx' else 'torchscript.pt'}"
            code = (generate_onnx_export(graph, out_model)
                    if args.command == "onnx"
                    else generate_torchscript_export(graph, out_model))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        from pathlib import Path

        script = Path(args.out) / f"{stem}_export_{'onnx' if args.command == 'onnx' else 'jit'}.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(code)
        print(script)
        if args.run:
            result = subprocess.run([sys.executable, str(script)], cwd=".")
            return result.returncode
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
