# Architecture

```
ai_made_easy/
├── core/                 pure Python, no Qt (enforced by tests/test_structure.py)
│   ├── spec.py           BlockDefinition / ParamSpec / PortSpec contracts
│   ├── registry.py       block registry (self-describing JSON schemas)
│   ├── graph.py          graph IR: nodes, edges, shape inference, validation
│   ├── lints.py          architecture, data and training design rules
│   ├── migrate.py        project-file schema migrations
│   ├── blocks/           model-flow blocks (one module per family)
│   ├── codegen/          PyTorch + Keras emitters, helpers, ONNX/TorchScript
│   ├── training/         dataset/preprocessing/loss/optimizer catalogs, spec,
│   │                     PyTorch + Keras training-script templates, metrics code
│   ├── classic/          scikit-learn / boosting estimators and pipeline codegen
│   ├── targets.py        the single export-target table (UI, CLI, MCP)
│   ├── api.py            headless facade used by the CLI, MCP server and web server
│   ├── paths.py          $AIME_HOME (default ~/.aime) and its sub-folders
│   ├── runs/             persistent run history: run.json, epochs.jsonl, artefacts
│   ├── runner/           RunManager: training subprocesses + event listeners
│   ├── sweeps.py         grid / random / Optuna sweeps over design parameters
│   ├── deploy/           serving packages (FastAPI + Dockerfile), exports, model registry
│   ├── importers/        PyTorch (torch.fx), ONNX and Keras importers (run in a subprocess)
│   ├── block_packs.py    versioned custom blocks and .aimeblocks packs
│   ├── data/             dataset profiles, data lints, fingerprints, split and
│   │                     augmentation previews
│   └── …                 summary, fixes, suggestions, model card, bundles
├── ui/                   PySide6 application
│   ├── context.py        composition root: builds and wires everything
│   ├── workbench.py      main window shell (layout only)
│   ├── actions_catalog.py  every action declared as data
│   ├── theme.py, icons.py  design system
│   ├── canvas/           the only package importing OdenGraphQt
│   ├── features/         panels and dialogs (presentation only)
│   └── services/         graph settle pipeline, processes, exports, projects
├── worker/               subprocess wrapper streaming JSON training events
├── mcp/                  MCP server over core/api.py
├── server/               FastAPI REST + WebSocket server and the built web UI (static/)
└── cli.py                headless command line
web/                      React + TypeScript + Vite + React Flow frontend (built into server/static)
```

## Design rules

| Rule | Enforced by |
|---|---|
| `core/` never imports Qt | `test_core_stays_qt_free` |
| OdenGraphQt is only imported in `ui/canvas/` | `test_odengraphqt_only_imported_inside_canvas_package` |
| Every action spec resolves to a context slot | `test_every_catalog_slot_resolves_on_context` |
| The workbench stays layout-only | `test_workbench_is_layout_only` |
| Every model block generates runnable PyTorch and Keras code with the designed output shape | `tests/test_codegen_runtime.py` |
| Every dataset modality trains end to end | `tests/test_training_e2e.py` |
| Every classic estimator fits end to end | `tests/test_classic.py` |
| Every block can be placed and edited on the canvas | `tests/test_ui.py` |
| Imported models reproduce the original outputs | `tests/test_import.py` |
| Serving packages answer `/predict` like the checkpoint | `tests/test_deploy.py` |
| The split preview equals the generated split | `tests/test_data.py` |
| Every REST endpoint and the run WebSocket | `tests/test_server.py` |
| The browser UI loads, validates and edits | `web/e2e/smoke.spec.ts` (Playwright) |

## Data flow

```
canvas edit ─► GraphService (debounced settle) ─► Graph IR ─► validate() + lints
      │                                                 ├─► Problems / card outlines / inspector
      │                                                 ├─► Summary, Code preview
      ▼                                                 └─► status bar
Train ─► training script (core/training or core/classic) ─► worker subprocess
      ─► epoch events ─► Training panel ─► run folder (checkpoint, metrics,
         predictions) ─► Error analysis · Saliency · Model card · Web demo
```

## Families, tasks and port roles

- `core/families.py`: every design belongs to one **family** (`neural`, `classic`,
  `llm`, ...). A family declares how it is detected, its validation rules (run
  after the generic graph checks), the frameworks that train it, its codegen
  targets and pip extras. Use `family_of(graph)` instead of block-prefix checks.
- `core/tasks.py`: a **task** (multiclass, regression, detection, ...) declares
  its target format, output role, losses, metrics, default metrics, trainer
  kind and serving schema. `TrainingSpec.task` resolves through it.
- Trainer kinds (`supervised`, `adversarial`, `diffusion`, `vae`, ...) select
  the training loop: `register_trainer(kind, renderer)` in
  `core/training/generate.py`.
- Ports carry a semantic `role` (`core.spec.ROLES`); `tensor` matches anything,
  two specific roles must agree. `multi=True` input ports take many wires.
- Blocks declare optional `requires` (modules) and `extra` (pip extra); missing
  modules are warnings with an install hint, shown in both libraries.
- `core/zoo/*.json`: pretrained catalogs generated offline by
  `scripts/build_zoo.py` (measured feature widths, parameters, accuracy,
  license).

## Vision tasks

`core/vision` adds detection, instance / semantic segmentation and keypoints
without touching the classification pipeline:

- `blocks.py`: task heads in the "Vision Tasks" category declare
  `meta["task"]` and a `meta["cost"]` hook (parameters, FLOPs) for the summary
  and budgets. Every wrapper (`helpers.py`) takes [0, 1] images and normalizes
  internally; detectors expose `losses()` / `detect()` and return padded
  detections `[D, 6]` from `forward`, so designed shapes still hold.
- `tasks.py`: registers the tasks (trainer kinds `detection` / `segmentation`)
  and a task resolver: a head block decides the task; a vision dataset plus a
  `[C, H, W]` output means semantic segmentation.
- `template.py`: the training script, registered with
  `register_trainer(kind, render, needs_spec=False)`, so it reads the graph
  directly instead of a `TrainingSpec`. It embeds `runtime.py` (dataset
  readers, metrics, RLE / overlays), which the app also executes for
  profiling and tests, so training and the Data workspace read data
  identically.
- `rules.py`: design rules registered with `lints.register_rule`. Tasks with
  their own training loop skip the classification-pipeline lints
  (`lints.PIPELINE_RULES`).
- `profile.py`: `register_profiler` hooks vision datasets into the Data
  workspace and data lints.

Loading a trained model sets `AIME_SKIP_PRETRAINED=1` so wrappers rebuild the
same architecture without downloading pretrained weights.

## Resource budgets

`core/budget.py` estimates a neural design's cost from the IR alone (no
torch import): `estimate(graph, device)` walks `model_nodes()` and gives each
layer its parameters, forward FLOPs and the activation values it keeps for the
backward pass. Linear and conv layers cost `2 × weights × output positions`,
recurrent and attention layers add their sequence terms, pretrained backbones
use the GMACs recorded in the model zoo. Views (flatten, reshape, ...) keep no
memory, and an output consumed only by ReLU / sigmoid / tanh / softmax is freed
because those activations save their own output.

Training memory = weights (+ fp16 copy under AMP) + gradients + optimizer state
(Adam 2, SGD-momentum 1, plain SGD 0 per weight) + activations × batch + the
device's runtime overhead. Latency (batch 1) = FLOPs at 35% of the device's peak
+ memory traffic over its bandwidth + a per-layer launch cost. Device profiles
are `core/devices.json` plus `~/.aime/devices.json`.

The project's budget is `graph.meta["budget"]` (device and limits; 0 = none).
`check()` compares the estimate with it; the `resource_budget` lint reports
overruns and `fixes.py` offers mixed precision or a fitting batch size with
gradient accumulation. The PyTorch training template prints a `resources:`
line after the first epoch (peak memory, step time); the worker turns it into
a `resources` event stored on the run record for calibration.

## Runs, sweeps and deployment

```
RunManager.start(graph) ─► RunHistory.create (design snapshot, params, data fingerprint)
      ─► training script in $AIME_HOME/runs/<run_id>/ ─► worker subprocess
      ─► env / epoch / log / done events ─► epochs.jsonl + listeners
         (desktop panels, MCP, WebSocket clients)
SweepRunner ─► for each trial: apply values ─► validate ─► RunManager.start(parent=sweep)
build_package(run dir) ─► app.py + model + inference state + Dockerfile (+ ONNX / TorchScript / Core ML)
ModelRegistry.register(run dir) ─► $AIME_HOME/models/<name>/<version>/ with a stage
```

Generated training scripts save `inference_state.pkl` (the preprocessing fitted
on the training split) and define `load_predictor()` / `infer()`, which the
serving package imports, so served predictions use exactly the training
pipeline.

## Data workspace

`core/data/profile.py` reads datasets the way the generated scripts do (same
readers, label encoding and folder order). `core/data/lints.py` turns profile
findings into Problems-panel warnings in the context of the pipeline,
suppressing what the design already handles. Profiles are cached by file
signature; the desktop computes them on a background thread
(`ui/services/data_service.py`). `core/data/splits.py` reproduces the generated
split exactly, and `core/data/augment.py` runs the design's torchvision
pipeline in the training environment.

## Web server

`server/app.py` exposes `core/api.py` over REST (`/api/...`) and streams run
events over `WS /api/runs/{id}/events`. It also serves the built frontend with
an SPA fallback. When `AIME_WEB_TOKEN` is set, every request needs the bearer
token. The frontend keeps the project as React Flow nodes / edges, converts to
the project JSON for every API call, and validates with a debounce.

## Intermediate representation

- Shapes are per sample, without the batch dimension, channels-first:
  `[F]`, sequences `[L, C]`, signals `[C, L]`, images `[C, H, W]`, volumes
  `[C, D, H, W]` — exactly PyTorch's conventions.
- The Keras emitter tracks a layout per tensor (channels-last for convolutional
  tensors, IR order for sequences) and inserts the minimal `Permute` layers.
- Blocks declare `shape_fn`, optional `param_fn` (parameter counts) and
  `checks_fn` (cross-parameter rules); configuration blocks carry `meta`
  describing modality, stage and task.

## Training scripts

Generated scripts are self-contained and readable:
data loading → split → preprocessing fitted on the training split → model →
loss / optimizer / scheduler → training loop with early stopping → test
evaluation → artefacts (`*_best.pt`, `metrics.json`, `predictions.json`,
`mistakes.json`, `classes.json`). Progress lines follow
`epoch e/E key=value …`, which the worker turns into JSON events.
