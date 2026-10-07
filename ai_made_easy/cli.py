"""Headless CLI over core — proves the app is drivable without the GUI.

This is the same JSON-in/JSON-out surface a future MCP agent will use.

Usage:
  aime blocks                       # list registered blocks (JSON schemas)
  aime validate <project.json>      # check a graph, print issues
  aime gen <project.json> -f pytorch [-o DIR]   # generate model code
  aime train <project.json> -f pytorch [-o DIR] # generate training script
  aime onnx <project.json> [-o DIR]   # generate (and with --run, execute) an ONNX export script
  aime jit <project.json> [-o DIR]    # generate (and with --run, execute) a TorchScript export script
  aime run <project.json> [-f auto]   # train headlessly (recorded in the run history)
  aime runs [list | show ID | compare ID ID… | delete ID]
  aime sweep <project.json> -p opt.lr=log:1e-4:1e-1 -p d1.units=int:16:128 \
             -p opt.nesterov=choice:true,false --metric accuracy --strategy tpe -n 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from ai_made_easy.core.codegen import FRAMEWORKS, export, export_training
from ai_made_easy.core.codegen import sanitize_identifier as sanitize_name
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aime", description="AI Made Easy — validate projects and generate code headlessly")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("blocks", help="list registered blocks")
    p_val = sub.add_parser("validate", help="validate a project graph")
    p_val.add_argument("project", help="path to project .json")
    for name, help_text, choices in (
            ("gen", "generate model code", FRAMEWORKS),
            ("train", "generate a training script (classic ML projects: sklearn)",
             (*FRAMEWORKS, "sklearn"))):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("project", help="path to project .json")
        p.add_argument("-f", "--framework", choices=choices, default="pytorch")
        p.add_argument("-o", "--out", default="exports", help="output directory")
    p_run = sub.add_parser("run", help="train headlessly and stream events")
    p_run.add_argument("project", help="path to project .json")
    p_run.add_argument("-f", "--framework", default="auto",
                       choices=("auto", "pytorch", "keras", "sklearn"))
    p_runs = sub.add_parser("runs", help="run history")
    p_runs.add_argument("action", nargs="?", default="list",
                        choices=("list", "show", "compare", "delete"))
    p_runs.add_argument("ids", nargs="*", help="run id(s)")
    p_runs.add_argument("--project", default=None, help="filter by project name")
    p_sw = sub.add_parser("sweep", help="hyperparameter sweep (trials go to the run history)")
    p_sw.add_argument("project", help="path to project .json")
    p_sw.add_argument("-p", "--param", action="append", default=[], metavar="NODE.PARAM=SPACE",
                      help="float:LOW:HIGH | log:LOW:HIGH | int:LOW:HIGH[:STEP] | "
                           "choice:A,B,C (repeatable)")
    p_sw.add_argument("--metric", default="val_loss")
    p_sw.add_argument("--direction", choices=("min", "max"), default="")
    p_sw.add_argument("--strategy", choices=("tpe", "random", "grid"), default="tpe")
    p_sw.add_argument("-n", "--trials", type=int, default=10)
    p_sw.add_argument("--points", type=int, default=3, help="grid points per range")
    p_sw.add_argument("--seed", type=int, default=0)
    p_sw.add_argument("--list-params", action="store_true",
                      help="print the parameters that can be swept and exit")
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

    if args.command == "runs":
        return _runs_command(args)

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

    if args.command == "sweep":
        return _sweep_command(args, graph)

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
        try:
            run_id = mgr.start(graph, args.framework)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
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


def parse_space(text: str, points: int = 3) -> dict:
    """``node.param=kind:...`` → a sweep dimension dict."""
    if "=" not in text or "." not in text.split("=", 1)[0]:
        raise ValueError(f"expected NODE.PARAM=SPACE, got {text!r}")
    key, space = text.split("=", 1)
    node, param = key.rsplit(".", 1)
    kind, _, rest = space.partition(":")
    if kind == "choice":
        def conv(v: str):
            low = v.strip().lower()
            if low in ("true", "false"):
                return low == "true"
            for cast in (int, float):
                try:
                    return cast(v)
                except ValueError:
                    pass
            return v.strip()

        return {"node": node, "param": param, "kind": "choice",
                "values": [conv(v) for v in rest.split(",") if v.strip()]}
    parts = rest.split(":")
    if kind in ("float", "log") and len(parts) == 2:
        return {"node": node, "param": param, "kind": "float", "low": float(parts[0]),
                "high": float(parts[1]), "log": kind == "log", "points": points}
    if kind == "int" and len(parts) in (2, 3):
        return {"node": node, "param": param, "kind": "int", "low": int(parts[0]),
                "high": int(parts[1]), "step": int(parts[2]) if len(parts) == 3 else 1}
    raise ValueError(f"cannot parse the search space {space!r}")


def _sweep_command(args, graph) -> int:  # noqa: ANN001
    import time as _time

    from ai_made_easy.core import api

    if args.list_params:
        for p in api.sweepable_params(graph)["params"]:
            space = (",".join(map(str, p["values"])) if p["kind"] == "choice"
                     else f"{p['low']}:{p['high']}")
            print(f"{p['key']:<32} {p['kind']:<7} current={p['current']!r:<10} {space}")
        return 0
    try:
        spec = {"dimensions": [parse_space(t, args.points) for t in args.param],
                "metric": args.metric, "direction": args.direction,
                "strategy": args.strategy, "max_trials": args.trials, "seed": args.seed}
        sweep_id = api.start_sweep(graph, spec)["sweep_id"]
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"type": "sweep_started", "sweep_id": sweep_id}), flush=True)
    reported = 0
    while True:
        record = api.get_sweep(sweep_id)
        done = [t for t in record["trials"] if t["state"] not in ("pending", "running")]
        for t in done[reported:]:
            print(json.dumps({"type": "trial", **t}), flush=True)
        reported = len(done)
        if record["state"] in ("finished", "stopped", "failed"):
            print(json.dumps({"type": "sweep_" + record["state"], "best": record["best"]}),
                  flush=True)
            return 0 if record["state"] == "finished" and record["best"] else 1
        _time.sleep(0.5)


def _runs_command(args) -> int:  # noqa: ANN001
    from ai_made_easy.core import api

    try:
        if args.action == "list":
            rows = api.list_runs(args.project)["runs"]
            for r in rows:
                metrics = " ".join(f"{k}={v:.4g}" for k, v in
                                   list(r["final_metrics"].items())[:4])
                print(f"{r['run_id']}  {r['status']:<9} {r['framework']:<8} "
                      f"{r['name']:<24} {metrics}")
            print(f"{len(rows)} run(s)")
            return 0
        if not args.ids:
            print("error: give a run id", file=sys.stderr)
            return 2
        if args.action == "show":
            print(json.dumps(api.get_run(args.ids[0]), indent=2, default=str))
        elif args.action == "compare":
            print(json.dumps(api.compare_runs(args.ids), indent=2, default=str))
        elif args.action == "delete":
            for run_id in args.ids:
                api.delete_run(run_id)
                print(f"deleted {run_id}")
        return 0
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
