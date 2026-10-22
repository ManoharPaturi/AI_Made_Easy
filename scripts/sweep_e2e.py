"""End-to-end sweep: every sample and every recipe loads, validates, generates code and
(with --train) trains a tiny run, then (with --deploy) builds a verified serving package.

    python scripts/sweep_e2e.py                    # load / validate / generate
    python scripts/sweep_e2e.py --train --deploy   # the nightly job: everything
    python scripts/sweep_e2e.py -k vae --train     # only cases whose name contains "vae"

Designs whose blocks need an extra that is not installed are reported as skipped, not
failed. Training settings are shrunk (one epoch, a few hundred RL steps, short MCMC
chains) so the sweep checks that each design *runs*, not how well it learns. Exits 1
when any case fails; --json writes the full report.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from ai_made_easy.core import api
from ai_made_easy.core.codegen import CodegenError
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import missing_requirements

# setting name -> the most a sweep run may use (only ever lowered, never raised)
# image and audio sets feed the heaviest models (Mask R-CNN, wav2vec2), so they shrink too
SHRINK = {"epochs": 1, "n_epochs": 1, "total_timesteps": 256, "eval_episodes": 1,
          "draws": 50, "tune": 50, "advi_iterations": 200, "n_estimators": 20,
          "sample_steps": 10, "gradient_steps": 1,
          "n_images": 64, "n_clips": 64}
RUN_TIMEOUT = 900.0
# checkpoints too big for a smoke run even online (Stable Diffusion is several GB)
HEAVY_WEIGHTS = {"gen.sd_lora"}
# blocks that fetch data over the network when a run starts (offline sweeps skip them)
NETWORK_DATA = {"data.torchvision", "data.huggingface", "data.ogb_nodes", "data.ogb_graphs",
                "data.ogb_links"}


@dataclass
class Case:
    name: str
    graph: dict
    stages: dict = field(default_factory=dict)   # stage -> "ok" | "skip: …" | "fail: …"
    seconds: float = 0.0

    @property
    def failed(self) -> bool:
        return any(v.startswith("fail") for v in self.stages.values())


def sample_cases() -> list[Case]:
    return [Case(f"sample:{n}", api.read_sample(n)) for n in api.list_samples()["samples"]]


def recipe_cases() -> list[Case]:
    """Each recipe on its demo data, once per task it trains (demo recipes only)."""
    from ai_made_easy.core import recipes

    out = []
    for recipe in recipes.all_recipes():
        for task in recipe.tasks:
            if recipe not in recipes.recipes_for(task):
                continue                         # needs the user's data: no demo design
            name = f"recipe:{recipe.id}/{task}"
            missing = recipe.missing()
            if missing:
                out.append(Case(name, {}, {"build": f"skip: needs {', '.join(missing)}"}))
                continue
            try:
                graph = recipes.build(recipe.id, task, recipes.detect(None), {}).to_dict()
            except Exception as exc:  # noqa: BLE001 — a recipe that cannot build is a bug
                out.append(Case(name, {}, {"build": f"fail: {exc}"}))
                continue
            out.append(Case(name, graph, {"build": "ok"}))
    return out


def _network_data(graph: Graph) -> list[str]:
    from ai_made_easy.core.training.data_catalog import SKLEARN

    found = []
    for node in graph.nodes.values():
        params = node.resolved_params()
        if node.type_id in NETWORK_DATA:
            found.append(node.type_id)
        elif node.type_id == "data.sklearn" and \
                SKLEARN.get(params.get("dataset"), ("",))[0].startswith("fetch_"):
            found.append(f"data.sklearn ({params['dataset']})")
    return found


def shrink(graph: dict, online: bool = False) -> dict:
    """A copy whose training settings are cut down to a smoke run (and, offline, whose
    pretrained blocks build with random weights: no checkpoint downloads)."""
    reg = get_registry()
    data = json.loads(json.dumps(graph))
    for node in data.get("nodes", []):
        params = node.get("params") or {}
        if (not online or node.get("type") in HEAVY_WEIGHTS) and reg.has(node.get("type", "")):
            spec = next((p for p in reg.get(node["type"]).params if p.name == "weights"), None)
            if spec is not None and "none" in (spec.options or ()):
                params["weights"] = "none"
        for key, cap in SHRINK.items():
            value = params.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and value > cap:
                params[key] = cap
    return data


def _missing(graph: Graph) -> list[str]:
    reg = get_registry()
    needs = set()
    for node in graph.nodes.values():
        if reg.has(node.type_id):
            needs.update(missing_requirements(reg.get(node.type_id)))
    return sorted(needs)


def check(case: Case, train: bool, deploy: bool, workdir: Path, online: bool) -> None:
    from ai_made_easy.core.families import family_of

    start = time.time()
    try:
        _check(case, train, deploy, workdir, family_of, online)
    except Exception:  # noqa: BLE001 — record and keep sweeping
        stage = next(iter(reversed(case.stages)), "load")
        case.stages[stage + "!"] = "fail: " + traceback.format_exc()[-2500:]
    case.seconds = round(time.time() - start, 1)


def _check(case: Case, train: bool, deploy: bool, workdir: Path, family_of,  # noqa: ANN001
           online: bool) -> None:
    if not case.graph:
        return
    graph = Graph.from_dict(case.graph)
    case.stages["load"] = "ok"
    missing = _missing(graph)
    if missing:
        case.stages["validate"] = f"skip: needs {', '.join(missing)}"
        return
    result = api.validate(graph)
    errors = [i["message"] for i in result["issues"] if i["severity"] == "error"]
    case.stages["validate"] = "ok" if not errors else "fail: " + "; ".join(errors[:3])
    if errors:
        return
    family = family_of(graph)
    for target in family.targets:
        try:
            api.generate(graph, target)
            case.stages[f"generate:{target}"] = "ok"
        except CodegenError as exc:   # a target the design cannot use, with the reason
            case.stages[f"generate:{target}"] = f"skip: {exc}"
        except Exception as exc:  # noqa: BLE001
            case.stages[f"generate:{target}"] = f"fail: {type(exc).__name__}: {exc}"
    if not train:
        return
    if not family.trainable:
        case.stages["train"] = "skip: not trainable in the app"
        return
    absent = [i["message"] for i in result["issues"] if "not installed" in i["message"]]
    if absent:          # the design itself warns that a package is missing
        case.stages["train"] = f"skip: {absent[0][:120]}"
        return
    network = _network_data(graph)
    if network and not online:
        case.stages["train"] = f"skip: downloads data ({', '.join(network)}): use --online"
        return
    small = Graph.from_dict(shrink(graph.to_dict(), online))
    if family.id == "pipeline":
        started = api.start_pipeline(small, project="sweep")
        record = api.wait_pipeline(started["pipeline_id"], RUN_TIMEOUT)
        failed = {k: v.get("message", "") for k, v in record["stages"].items()
                  if v.get("state") != "finished"}
        case.stages["train"] = "ok" if record["state"] == "finished" and not failed \
            else f"fail: {record['state']}: {failed}"
        return
    run_id = api.start_training(small, project="sweep")["run_id"]
    try:
        status = api.manager().wait(run_id, RUN_TIMEOUT)
    except TimeoutError:
        api.stop_run(run_id)              # never leave a worker running behind the sweep
        case.stages["train"] = f"fail: still running after {RUN_TIMEOUT:.0f}s (stopped)"
        return
    rec = api.manager().history.get(run_id)
    if status["state"] != "finished":
        case.stages["train"] = f"fail: {status['state']}: {(rec.error or '')[-1500:]}"
        return
    case.stages["train"] = "ok"
    if deploy:
        out = workdir / run_id
        result = api.deploy_run(run_id, str(out))
        case.stages["deploy"] = "ok" if result.get("verified") \
            else f"fail: package not verified: {result.get('log', '')[-1500:]}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", action="store_true", help="train a tiny run of each design")
    ap.add_argument("--deploy", action="store_true", help="also build a serving package")
    ap.add_argument("--online", action="store_true",
                    help="allow dataset and pretrained-weight downloads (the nightly job)")
    ap.add_argument("-k", default="", help="only cases whose name contains this")
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--no-recipes", action="store_true")
    ap.add_argument("--no-samples", action="store_true")
    args = ap.parse_args(argv)

    cases = ([] if args.no_samples else sample_cases()) + \
            ([] if args.no_recipes else recipe_cases())
    cases = [c for c in cases if args.k in c.name]
    with tempfile.TemporaryDirectory(prefix="aime-sweep-") as tmp:
        if args.train:
            from ai_made_easy.core.runner.manager import RunManager
            from ai_made_easy.core.runs.history import RunHistory

            api.set_manager(RunManager(RunHistory(Path(tmp) / "runs")))
        for case in cases:
            check(case, args.train, args.deploy, Path(tmp), args.online)
            mark = "FAIL" if case.failed else "ok  "
            skipped = [v for v in case.stages.values() if v.startswith("skip")]
            note = f"  ({skipped[-1][:100]})" if skipped and not case.failed else ""
            print(f"{mark} {case.name}  {case.seconds}s{note}", flush=True)
            for stage, verdict in case.stages.items():
                if verdict.startswith("fail"):
                    print(f"     {stage}: {verdict.strip()[:2000]}", flush=True)
    failed = [c for c in cases if c.failed]
    print(f"\n{len(cases) - len(failed)}/{len(cases)} cases passed")
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"name": c.name, "stages": c.stages, "seconds": c.seconds} for c in cases],
            indent=1), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
