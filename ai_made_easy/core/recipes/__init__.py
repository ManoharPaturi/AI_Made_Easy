"""Recipes: parameterized design templates per task, data kind and size tier.

A :class:`Recipe` builds a complete design (data, preprocessing, model, loss, optimizer,
trainer, metrics) from :class:`~ai_made_easy.core.recipes.facts.DataFacts` and a few knobs
(learning rate, width, depth, ...). :func:`recommend` ranks the recipes that fit a task,
the user's data and the project's budget, each with the reasons it was ranked there, a
validated graph and its cost estimate. :func:`build` makes one design; every block it
adds carries a short note on why it is there, which :func:`explain` returns for the
"Explain this design" panel. AutoML (:mod:`ai_made_easy.core.automl`) searches over the
recipes and their knobs.

Pure Python, Qt-free.
"""
from __future__ import annotations

import copy
import importlib.util
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_made_easy.core.recipes.facts import DataFacts, demo_facts, detect

__all__ = ["TIERS", "DataFacts", "Draft", "Knob", "Recipe", "RecipeError", "Suggestion",
           "all_recipes", "autofit", "build", "demo_facts", "detect", "explain", "get_recipe",
           "modalities_for", "recipes_for", "recommend", "register_recipe"]

TIERS = ("small", "medium", "large", "pretrained")
TIER_LABELS = {"small": "Small", "medium": "Medium", "large": "Large",
               "pretrained": "Pretrained"}


class RecipeError(ValueError):
    pass


# ===================================================================== knobs

@dataclass(frozen=True)
class Knob:
    """A setting a recipe exposes to the user and to AutoML."""
    name: str
    kind: str                     # float | int | choice
    default: Any
    low: float | None = None
    high: float | None = None
    log: bool = False
    values: tuple = ()
    help: str = ""
    target: str = ""              # "node.param" the value is written to (starter recipes)

    def dimension(self):  # noqa: ANN201 — core.sweeps.Dimension
        from ai_made_easy.core.sweeps import Dimension

        return Dimension(node="recipe", param=self.name, kind=self.kind, low=self.low,
                         high=self.high, log=self.log, values=list(self.values))

    def clean(self, value: Any) -> Any:
        if value is None:
            return self.default
        if self.kind == "choice":
            if value not in self.values:
                raise RecipeError(f"{self.name} must be one of {list(self.values)}")
            return value
        number = float(value)
        if self.low is not None and number < self.low or self.high is not None \
                and number > self.high:
            raise RecipeError(f"{self.name} must be within [{self.low}, {self.high}]")
        return int(round(number)) if self.kind == "int" else number

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "default": self.default, "low": self.low,
                "high": self.high, "log": self.log, "values": list(self.values),
                "help": self.help}


LR = Knob("lr", "float", 1e-3, 1e-5, 0.1, log=True, help="Optimizer learning rate")


# ===================================================================== drafts

class Draft:
    """A design under construction: blocks, wires and why each block is there."""

    def __init__(self, name: str, title: str = "", description: str = ""):
        self.name, self.title, self.description = name, title, description
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.why: dict[str, str] = {}
        self._chain_y = 0
        self._side_y = 0

    def _id(self, type_id: str) -> str:
        base = type_id.split(".", 1)[-1]
        used = {n["id"] for n in self.nodes}
        nid, k = base, 1
        while nid in used:
            k += 1
            nid = f"{base}_{k}"
        return nid

    def add(self, type_id: str, why: str = "", *, nid: str | None = None,
            **params: Any) -> str:
        """A configuration block (data, preprocessing, loss, optimizer, trainer, metric)."""
        nid = nid or self._id(type_id)
        self.nodes.append({"id": nid, "type": type_id, "params": params,
                           "position": [360, self._side_y]})
        self._side_y += 80
        if why:
            self.why[nid] = why
        return nid

    def chain(self, *steps: tuple) -> list[str]:
        """Blocks wired one after another: ``(type_id, params, why)`` per step."""
        ids = []
        for step in steps:
            type_id, params, why = (tuple(step) + ({}, ""))[:3]
            nid = self._id(type_id)
            self.nodes.append({"id": nid, "type": type_id, "params": dict(params or {}),
                               "position": [0, self._chain_y]})
            self._chain_y += 90
            if why:
                self.why[nid] = why
            if ids:
                self.edges.append({"from": f"{ids[-1]}/out", "to": f"{nid}/in"})
            ids.append(nid)
        return ids

    def to_graph(self) -> dict:
        return {"schema_version": 1, "name": self.name, "nodes": self.nodes,
                "edges": self.edges, "meta": {"title": self.title,
                                              "description": self.description}}


@dataclass
class Context:
    """What a recipe's builder sees."""
    task: str
    facts: DataFacts
    knobs: dict[str, Any]

    def __getitem__(self, name: str) -> Any:
        return self.knobs[name]


# ===================================================================== recipes

@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    tasks: tuple[str, ...]
    tier: str
    modality: str
    description: str
    builder: Callable[[Context], Draft]
    data_kinds: tuple[str, ...] = ("demo",)
    demo_tasks: tuple[str, ...] = ()      # tasks the demo data supports (empty: all)
    knobs: tuple[Knob, ...] = ()
    requires: tuple[str, ...] = ()        # importable modules
    extra: str = ""                       # pip extra that installs them
    downloads: str = ""                   # fetched on first use (demo data, weights)
    family: str = "neural"
    rows: tuple[int, int] = (0, 10**12)   # data sizes the recipe suits best
    strengths: str = ""                   # one line shown as a reason
    priority: float = 0.0                 # tie-break within a tier

    def missing(self) -> list[str]:
        return [m for m in self.requires if importlib.util.find_spec(m) is None]

    @property
    def installed(self) -> bool:
        return not self.missing()

    def knob(self, name: str) -> Knob:
        for k in self.knobs:
            if k.name == name:
                return k
        raise RecipeError(f"{self.id} has no setting {name!r}")

    def settings(self, values: dict | None = None) -> dict[str, Any]:
        values = dict(values or {})
        unknown = set(values) - {k.name for k in self.knobs}
        if unknown:
            raise RecipeError(f"{self.id} has no setting(s) {sorted(unknown)}")
        return {k.name: k.clean(values.get(k.name)) for k in self.knobs}

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "tasks": list(self.tasks),
                "tier": self.tier, "tier_label": TIER_LABELS[self.tier],
                "modality": self.modality, "description": self.description,
                "data_kinds": list(self.data_kinds), "knobs": [k.to_dict() for k in self.knobs],
                "requires": list(self.requires), "extra": self.extra, "missing": self.missing(),
                "downloads": self.downloads, "family": self.family,
                "strengths": self.strengths}


_RECIPES: dict[str, Recipe] = {}
_LOADED = False


def register_recipe(recipe: Recipe) -> Recipe:
    if recipe.tier not in TIERS:
        raise RecipeError(f"{recipe.id}: tier must be one of {TIERS}")
    _RECIPES[recipe.id] = recipe
    return recipe


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    import ai_made_easy.core.blocks  # noqa: F401 — families register their tasks
    from ai_made_easy.core.recipes import media, starters, tabular  # noqa: F401
    from ai_made_easy.core.registry import get_registry

    get_registry()


def all_recipes() -> list[Recipe]:
    _load()
    return list(_RECIPES.values())


def get_recipe(recipe_id: str) -> Recipe:
    _load()
    if recipe_id not in _RECIPES:
        raise RecipeError(f"no recipe {recipe_id!r}")
    return _RECIPES[recipe_id]


def recipes_for(task: str, facts: DataFacts | None = None,
                modality: str | None = None) -> list[Recipe]:
    """Recipes that train ``task`` on data like ``facts`` (demo data when None)."""
    kind = (facts or demo_facts()).kind
    return [r for r in all_recipes() if task in r.tasks and kind in r.data_kinds
            and (kind != "demo" or not r.demo_tasks or task in r.demo_tasks)
            and (not modality or r.modality == modality)]


def modalities_for(task: str) -> list[str]:
    """Data modalities that have demo recipes for ``task`` (the wizard asks which)."""
    return list(dict.fromkeys(r.modality for r in recipes_for(task)))


# ===================================================================== building

SAFE_FIXES = re.compile(r"Set the Input shape|produces \d+ features per sample|[Ss]et \w+ to "
                        r"|Use the data's categories|Training needs about|Normalize the table")
# suggestions (info) that size the design to the data
INFO_FIXES = re.compile(r"to save memory|Set the Input shape")


def autofit(graph):  # noqa: ANN001, ANN201 — Graph -> (Graph, [applied fix descriptions])
    """Apply the safe Quick Fixes (input width, sizes the data dictates, categories,
    memory) until none is left; returns the fitted graph and what was changed."""
    from ai_made_easy.core.fixes import fix_for_issue

    applied: list[str] = []
    tried: set[str] = set()
    for _ in range(10):
        issues = [i for i in graph.validate() if i.message not in tried and (
            i.severity in ("error", "warning") and SAFE_FIXES.search(i.message)
            or i.severity == "info" and INFO_FIXES.search(i.message))]
        fixed = None
        for issue in issues:
            tried.add(issue.message)
            fixed = fix_for_issue(graph, issue)
            if fixed is not None:
                break
        if fixed is None:
            break
        _label, description, graph = fixed
        applied.append(description)
    return graph, applied


def build(recipe: Recipe | str, task: str | None = None, facts: DataFacts | None = None,
          knobs: dict | None = None, budget: dict | None = None):  # noqa: ANN201 — Graph
    """The recipe's design for ``task`` on ``facts``, fitted to the data and the budget."""
    from ai_made_easy.core import budget as budget_mod
    from ai_made_easy.core.graph import Graph

    recipe = get_recipe(recipe) if isinstance(recipe, str) else recipe
    task = task or recipe.tasks[0]
    if task not in recipe.tasks:
        raise RecipeError(f"{recipe.title} does not train {task!r} (it trains "
                          f"{', '.join(recipe.tasks)})")
    facts = facts or demo_facts()
    if facts.kind not in recipe.data_kinds:
        raise RecipeError(f"{recipe.title} cannot read {facts.kind.replace('_', ' ')} data")
    settings = recipe.settings(knobs)
    draft = recipe.builder(Context(task, copy.deepcopy(facts), settings))
    graph = Graph.from_dict(draft.to_graph())
    for knob in recipe.knobs:
        if knob.target:
            node_id, param = knob.target.rsplit(".", 1)
            if node_id in graph.nodes:
                graph.nodes[node_id].params[param] = settings[knob.name]
    if budget:
        budget_mod.set_budget(graph, **{k: v for k, v in budget.items() if v})
    graph, applied = autofit(graph)
    graph.meta = {**(graph.meta or {}), "recipe": {
        "id": recipe.id, "title": recipe.title, "tier": recipe.tier, "task": task,
        "knobs": settings, "data": facts.kind, "why": draft.why, "adapted": applied}}
    return graph


# ===================================================================== ranking

@dataclass
class Suggestion:
    recipe: Recipe
    task: str
    score: float
    graph: dict | None = None
    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    estimate: dict = field(default_factory=dict)
    over_budget: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.recipe.installed and not self.errors and not self.over_budget

    def to_dict(self, graph: bool = True) -> dict:
        out = {"recipe": self.recipe.to_dict(), "task": self.task, "score": round(self.score, 3),
               "reasons": self.reasons, "cautions": self.cautions, "errors": self.errors,
               "estimate": self.estimate, "over_budget": self.over_budget,
               "ready": self.ready}
        if graph:
            out["graph"] = self.graph
        return out


def _size_fit(recipe: Recipe, rows: int, demo: bool) -> tuple[float, str]:
    if demo:
        return {"small": 1.0, "medium": 0.8, "large": 0.4, "pretrained": 0.6}[recipe.tier], ""
    lo, hi = recipe.rows
    if rows < lo:
        return -1.0, f"may overfit {rows:,} rows (suits {lo:,}+)"
    if rows > hi:
        return 0.0, f"{rows:,} rows is more than it needs; a larger model may do better"
    return 1.5, f"suits {rows:,} rows"


def _estimate(graph) -> tuple[dict, list[str]]:  # noqa: ANN001
    from ai_made_easy.core import budget

    try:
        est = budget.estimate(graph)
        checks = budget.check(graph, est)
    except Exception:  # noqa: BLE001 — not every family has a cost model
        return {}, []
    row = {"params": est.params, "flops": est.flops,
           "train_memory_gb": round(est.train_memory_bytes() / 2**30, 3),
           "latency_ms": est.latency_ms() if est.device else None}
    over = [f"{c.kind.replace('_', ' ')} {c.used:.3g} {c.unit} > {c.limit:g} {c.unit}"
            for c in checks if c.over]
    return row, over


def recommend(task: str, facts: DataFacts | None = None, budget: dict | None = None, *,
              modality: str | None = None, knobs: dict[str, dict] | None = None,
              limit: int | None = None) -> list[Suggestion]:
    """Recipes for ``task`` ranked for the data and the budget, best first."""
    from ai_made_easy.core.tasks import get_task

    get_task(task)  # raises KeyError for an unknown task
    facts = facts or demo_facts()
    out: list[Suggestion] = []
    for recipe in recipes_for(task, facts, modality):
        score, size_note = _size_fit(recipe, facts.rows, facts.demo)
        score += recipe.priority
        sug = Suggestion(recipe, task, score)
        if recipe.strengths:
            sug.reasons.append(recipe.strengths)
        if size_note:
            (sug.reasons if score > 0.5 else sug.cautions).append(size_note)
        missing = recipe.missing()
        if missing:
            sug.score -= 5
            sug.cautions.append(f"needs {', '.join(missing)}: pip install "
                                f"'ai-made-easy[{recipe.extra}]'" if recipe.extra else
                                f"needs {', '.join(missing)}")
        if recipe.downloads:
            sug.cautions.append(f"downloads {recipe.downloads} on first use")
        try:
            graph = build(recipe, task, facts, (knobs or {}).get(recipe.id), budget)
        except Exception as exc:  # noqa: BLE001 — a recipe that cannot build is reported
            sug.score -= 20
            sug.errors.append(str(exc))
            out.append(sug)
            continue
        sug.graph = graph.to_dict()
        if not missing:
            sug.errors = [i.message for i in graph.validate() if i.severity == "error"]
            sug.score -= 10 * bool(sug.errors)
        sug.estimate, sug.over_budget = _estimate(graph)
        sug.score -= 4 * len(sug.over_budget)
        if budget and sug.estimate and not sug.over_budget:
            sug.reasons.append("fits the budget")
        adapted = graph.meta["recipe"]["adapted"]
        if adapted:
            sug.reasons.append("adapted to the data: " + "; ".join(adapted[:3]))
        out.append(sug)
    out.sort(key=lambda s: (-s.score, TIERS.index(s.recipe.tier), s.recipe.id))
    return out[:limit] if limit else out


# ===================================================================== explaining

def explain(graph) -> dict:  # noqa: ANN001 — Graph
    """Why each block is in the design: the recipe's notes, else the block's own
    description."""
    meta = (graph.meta or {}).get("recipe") or {}
    why = meta.get("why") or {}
    blocks = []
    for node in graph.nodes.values():
        definition = node.definition()
        note = why.get(node.instance_id)
        blocks.append({"id": node.instance_id, "type": node.type_id,
                       "name": definition.display_name, "category": definition.category,
                       "why": note or definition.description,
                       "source": "recipe" if note else "block"})
    recipe = None
    if meta.get("id"):
        try:
            r = get_recipe(meta["id"])
            recipe = {**{k: meta.get(k) for k in ("id", "title", "tier", "task", "knobs",
                                                  "adapted")},
                      "description": r.description, "strengths": r.strengths}
        except RecipeError:
            recipe = {k: meta.get(k) for k in ("id", "title", "tier", "task", "knobs")}
    from ai_made_easy.core.tasks import task_of

    task = task_of(graph)
    return {"recipe": recipe, "task": {"id": task.id, "label": task.label,
                                       "description": task.description} if task else None,
            "blocks": blocks}


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
