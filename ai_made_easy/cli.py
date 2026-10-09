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
  aime deploy RUN_ID -o DIR [--formats onnx,torchscript]   # model server package
  aime models [list | register RUN_ID NAME | stage NAME VERSION STAGE | deploy NAME VERSION -o DIR]
  aime serve DIR [--port 8000]       # run a deployment package locally
  aime import pytorch model.py --attr Net --shape 3,32,32 -o project.json
  aime import onnx model.onnx -o project.json      (also: keras model.keras)
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
    for name, help_text, choices, default in (
            ("gen", "generate model code", FRAMEWORKS, "pytorch"),
            ("train", "generate a training script (auto: the design's framework)", None,
             "auto")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("project", help="path to project .json")
        p.add_argument("-f", "--framework", choices=choices, default=default,
                       help="pytorch, keras, sklearn, pgmpy, pymc, ... (train: auto)")
        p.add_argument("-o", "--out", default="exports", help="output directory")
    p_run = sub.add_parser("run", help="train headlessly and stream events")
    p_run.add_argument("project", help="path to project .json")
    p_run.add_argument("-f", "--framework", default="auto",
                       help="auto (the design's framework), pytorch, keras, sklearn, ...")
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
    p_dep = sub.add_parser("deploy", help="build a model-server package from a finished run")
    p_dep.add_argument("run_id")
    p_dep.add_argument("-o", "--out", required=True, help="output folder (must be empty)")
    p_dep.add_argument("--formats", default="", help="comma-separated: onnx, onnx_int8, "
                       "torchscript, coreml (PyTorch); onnx, saved_model, tflite (Keras)")
    p_dep.add_argument("--name", default=None)
    p_mod = sub.add_parser("models", help="model registry")
    p_mod.add_argument("action", nargs="?", default="list",
                       choices=("list", "register", "stage", "deploy", "delete"))
    p_mod.add_argument("args", nargs="*")
    p_mod.add_argument("-o", "--out", default=None)
    p_mod.add_argument("--formats", default="")
    p_srv = sub.add_parser("serve", help="serve a deployment package with uvicorn")
    p_srv.add_argument("package")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_imp = sub.add_parser("import", help="import a PyTorch / ONNX / Keras model as a project")
    p_imp.add_argument("kind", choices=("pytorch", "onnx", "keras"))
    p_imp.add_argument("source", help=".py file or module (pytorch), .onnx, .keras / .h5")
    p_imp.add_argument("--attr", default="", help="model class or factory (pytorch)")
    p_imp.add_argument("--shape", default="", help="per-sample input shape, e.g. 3,224,224")
    p_imp.add_argument("--dtype", default="float32", choices=("float32", "int64"))
    p_imp.add_argument("--kwargs", default="{}", help="constructor arguments as JSON")
    p_imp.add_argument("-o", "--out", required=True, help="project .json to write")
    p_data = sub.add_parser("data", help="profile datasets, preview splits and augmentation")
    p_data.add_argument("action", choices=("profile", "check", "split", "augment",
                                           "fingerprint"))
    p_data.add_argument("source", help="project .json, or a data file / folder (profile)")
    p_data.add_argument("--target", default=None, help="target column (profiling a table)")
    p_data.add_argument("--json", action="store_true", help="print JSON")
    p_data.add_argument("-o", "--out", default="augmentation_preview",
                        help="output folder (augment)")
    p_web = sub.add_parser("web", help="serve the browser UI and REST API")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8765)
    p_web.add_argument("--open", action="store_true", help="open the browser")
    p_sum = sub.add_parser("summary", help="print the analytic model summary as JSON")
    p_sum.add_argument("project", help="path to project .json")
    p_bud = sub.add_parser("budget", help="estimate FLOPs / memory / latency on a device")
    p_bud.add_argument("project", nargs="?", default="", help="path to project .json")
    p_bud.add_argument("--device", default=None, help="device profile id (see --list)")
    p_bud.add_argument("--list", action="store_true", help="list device profiles")
    p_bud.add_argument("--json", action="store_true", help="print JSON")
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
    if args.command in ("deploy", "models", "serve"):
        return _deploy_command(args)
    if args.command == "import":
        return _import_command(args)
    if args.command == "data":
        return _data_command(args)
    if args.command == "web":
        try:
            from ai_made_easy.server.app import main as serve_web
        except ImportError as exc:
            print(f"error: {exc} — pip install 'ai-made-easy[web]'", file=sys.stderr)
            return 1
        serve_web(args.host, args.port, args.open)
        return 0

    if args.command == "budget" and (args.list or not args.project):
        return _budget_command(args, None)

    with open(args.project) as fh:
        graph = Graph.from_dict(json.load(fh))

    if args.command == "validate":
        from ai_made_easy.core.data.lints import data_issues, project_base

        issues = graph.validate() + data_issues(graph, base=project_base(args.project))
        for issue in issues:
            print(issue)
        print(f"{len(issues)} issue(s)")
        return 1 if any(i.severity == "error" for i in issues) else 0

    if args.command in ("gen", "train"):
        try:
            if args.command == "train":
                from ai_made_easy.core.runner.manager import resolve_framework

                args.framework = resolve_framework(graph, args.framework)
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

    if args.command == "budget":
        return _budget_command(args, graph)

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


def _budget_command(args, graph) -> int:  # noqa: ANN001
    from ai_made_easy.core import api, budget

    if graph is None:
        if not args.list:
            print("error: give a project, or --list to see device profiles", file=sys.stderr)
            return 2
        for d in api.list_devices()["devices"]:
            print(f"{d['id']:<18} {d['label']:<40} {d['memory_gb']:>5g} GB  "
                  f"{d['tflops_fp32']:>6g} TFLOPs fp32")
        return 0
    try:
        report = api.estimate_budget(graph, args.device)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        est = report["estimate"]
        device = est["device"]["label"] if est["device"] else "no device (use --device)"
        print(f"device            {device}")
        print(f"parameters        {est['params']:,} ({est['trainable_params']:,} trainable)")
        print(f"forward FLOPs     {budget.human_flops(est['flops'])} per sample")
        print(f"model size        {est['model_mb']:.1f} MB")
        print(f"training memory   {est['train_memory_gb']:.2f} GB at batch {est['batch_size']}"
              + (" (mixed precision)" if est["mixed_precision"] else ""))
        if est["latency_ms"] is not None:
            print(f"latency           {est['latency_ms']:.2f} ms per sample")
            print(f"training step     {est['step_time_ms']:.1f} ms")
        for c in report["checks"]:
            mark = "OVER" if c["over"] else "ok"
            print(f"budget {c['kind']:<13}{c['used']:.2f} / {c['limit']:g} {c['unit']}  {mark}")
    return 1 if any(c["over"] for c in report["checks"]) else 0


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


def _import_command(args) -> int:  # noqa: ANN001
    from pathlib import Path

    from ai_made_easy.core import api

    try:
        shape = [int(d) for d in args.shape.replace("x", ",").split(",") if d.strip()] or None
        result = api.import_model(args.kind, args.source, attr=args.attr, input_shape=shape,
                                  dtype=args.dtype, kwargs=json.loads(args.kwargs))
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["summary"])
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    if result["unsupported"]:
        return 1
    Path(args.out).write_text(json.dumps(result["graph"], indent=1))
    print(f"project: {args.out}")
    return 0 if result["ok"] else 1


def _data_command(args) -> int:  # noqa: ANN001
    from pathlib import Path

    from ai_made_easy.core import api
    from ai_made_easy.core.data.lints import project_base

    source = Path(args.source).expanduser()
    is_project = source.suffix == ".json" and source.is_file() and \
        "nodes" in json.loads(source.read_text() or "{}")
    try:
        if args.action == "profile" and not args.json:
            from ai_made_easy.core.data.lints import dataset_nodes
            from ai_made_easy.core.data.profile import profile_dataset, profile_path

            if not is_project:
                print(profile_path(source, target=args.target).text())
                return 0
            nodes = dataset_nodes(Graph.from_dict(json.loads(source.read_text())))
            if not nodes:
                print("error: the design has no dataset block", file=sys.stderr)
                return 1
            print(profile_dataset(nodes[0][1], nodes[0][2], project_base(source)).text())
            return 0
        if args.action == "profile" and not is_project:
            result = api.profile_data(path=str(source), target=args.target)
        else:
            if not is_project:
                print("error: give a project .json", file=sys.stderr)
                return 2
            graph = json.loads(source.read_text())
            base = str(project_base(source))
            result = {
                "profile": lambda: api.profile_data(graph, base=base),
                "check": lambda: api.data_issues(graph, base=base),
                "split": lambda: api.split_preview(graph, base=base),
                "fingerprint": lambda: api.data_fingerprint(graph, base=base),
                "augment": lambda: api.augmentation_preview(graph, args.out, base=base),
            }[args.action]()
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json or args.action == "profile":
        print(json.dumps(result, indent=2, default=str))
    elif args.action == "check":
        for issue in result["issues"]:
            print(f"{issue['severity']}: {issue['message']}")
        print(f"{len(result['issues'])} data issue(s)")
    elif args.action == "split":
        print(f"split: {result['method']}")
        print("            " + "".join(f"{s:>9}" for s in ("train", "val", "test")))
        for cls, counts in result["per_class"].items():
            print(f"{cls[:12]:<12}" + "".join(f"{counts[s]:>9,}" for s in
                                                ("train", "val", "test")))
        print(f"{'total':<12}" + "".join(f"{result['totals'][s]:>9,}" for s in
                                         ("train", "val", "test")))
        for note in result["notes"]:
            print(f"note: {note}")
    elif args.action == "augment":
        for item in result["images"]:
            print(f"{item['source']}: eval {item['eval']}, {len(item['train'])} training views")
        print(f"preview images in {args.out}")
    else:
        print(result["fingerprint"] or "(no local data)")
    return 0


def _deploy_command(args) -> int:  # noqa: ANN001
    from ai_made_easy.core import api

    formats = [f.strip() for f in getattr(args, "formats", "").split(",") if f.strip()]
    try:
        if args.command == "serve":
            result = subprocess.run([sys.executable, "-m", "uvicorn", "app:app", "--host",
                                     args.host, "--port", str(args.port)], cwd=args.package)
            return result.returncode
        if args.command == "deploy":
            result = api.deploy_run(args.run_id, args.out, formats, args.name)
            print(result["log"].replace("EXPORT-DONE", "").strip())
            print(f"package: {result['path']}")
            return 0
        a = args.args
        if args.action == "list":
            for m in api.list_models()["models"]:
                metrics = " ".join(f"{k}={v:.4g}" for k, v in list(m["metrics"].items())[:3]
                                   if isinstance(v, (int, float)))
                print(f"{m['name']:<24} v{m['version']:<4} {m['stage']:<11} "
                      f"{m['framework']:<8} {metrics}")
        elif args.action == "register" and len(a) == 2:
            m = api.register_model(a[0], a[1])
            print(f"registered {m['name']} v{m['version']}")
        elif args.action == "stage" and len(a) == 3:
            m = api.set_model_stage(a[0], int(a[1]), a[2])
            print(f"{m['name']} v{m['version']} -> {m['stage']}")
        elif args.action == "deploy" and len(a) in (1, 2) and args.out:
            result = api.deploy_model(a[0], a[1] if len(a) == 2 else "latest", args.out,
                                      formats)
            print(f"package: {result['path']}")
        elif args.action == "delete" and len(a) == 2:
            api.delete_model(a[0], int(a[1]))
            print(f"deleted {a[0]} v{a[1]}")
        else:
            print("usage: aime models [list | register RUN_ID NAME | stage NAME VERSION "
                  "STAGE | deploy NAME [VERSION] -o DIR | delete NAME VERSION]",
                  file=sys.stderr)
            return 2
        return 0
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


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
