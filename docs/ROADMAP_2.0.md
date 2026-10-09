# AI Made Easy 2.0 — major update plan

> **Status: complete — released as 2.0.0.**
>
> | Phase | Delivered in |
> |---|---|
> | 0 Headless foundations | `core/api.py`, `core/runs/`, `core/paths.py` |
> | 1 Experiments & tuning | `core/sweeps.py`, Experiments dock |
> | 2 Deploy & serve | `core/deploy/`, Models tab, `aime deploy / serve / models` |
> | 3 Import & round-trip | `core/importers/`, `core/block_packs.py` |
> | 4 Data workspace | `core/data/`, Data dock |
> | 5 Web version | `ai_made_easy/server/`, `web/`, `Dockerfile` |
> | 6 Release 2.0.0 | docs, CHANGELOG, CI (web, docker, package jobs) |
>
> Differences from the plan: the run store landed as `core/runs/history.py`
> (`$AIME_HOME/runs/<run_id>/`, not per-project folders); TFLite export was
> left out (Core ML, ONNX and TorchScript cover the targets); the project
> schema did not change, so no migration was needed.

## Context
1.0 (main @ f842216) is a production desktop designer: 345 blocks, live validation,
verified PyTorch / Keras / sklearn codegen, in-app training. The user wants a major
update covering all four areas — **Deploy & serve, Experiments & tuning, Import &
round-trip, Data workspace** — and a **web version** (browser UI + backend) so it can be
hosted later. Survey findings that shape the plan:
- Runs live in `mkdtemp` folders; no history, no config snapshot, no comparison.
- `core/runner/manager.py` RunManager is Qt-free but PyTorch-only; UI uses its own
  `ui/services/process_service.py`.
- Web demo / ONNX / TorchScript logic is partly in `ui/context.py`.
- Hyperparameter search exists only for sklearn (inside the generated script).
- `mcp/server.py` `_impl_*` functions are the de-facto headless API.

Workflow (unchanged): one branch per phase step → `git merge --no-ff` → push main;
full pytest + ruff green before every merge; CI must stay green.

---

## Phase 0 — Headless foundations
- `core/api.py`: one facade (blocks, validate, generate, summary, expand, samples,
  runs, sweeps, deploy, import, data) — MCP `_impl_*`, CLI and the web server call it.
- `core/runs/store.py`: persistent **RunStore on disk** `~/.aime/runs/<project>/<run_id>/`
  with `run.json` (graph snapshot, framework, params, seed, env versions, data
  fingerprint, status, timings), `epochs.jsonl` (every epoch event), artifacts.
- RunManager: pytorch + keras + sklearn, writes through the run store; UI
  ProcessService delegates workdir creation to it.
- Move web-demo builder call + runtime export orchestration into `core/`.

## Phase 1 — Experiments & tuning
- **Runs dock** (UI): history table (status, metrics, duration, tags), multi-select →
  compare table + overlaid curves, diff of graph params between runs, restore a run's
  graph, delete/rename/tag, open folder.
- **Sweeps** `core/sweeps/`: spec = list of (node_id, param, distribution) +
  objective metric + direction + budget; strategies grid / random / Optuna TPE
  (optional dep, falls back to random); trials executed sequentially through RunManager
  as child runs; early-stop failed trials; leaderboard; "Apply best params".
- Sweep dialog: pick any numeric/choice param from the graph (incl. trainer lr, batch
  size, dropout p, hidden units); constraints validated per trial via `Graph.validate`
  (invalid configs skipped, not run).
- Reproducibility: environment capture, deterministic flag in trainer.
- CLI `aime sweep`, `aime runs`; MCP tools.

## Phase 2 — Deploy & serve
- `core/deploy/`: from a finished run build a **serving package**:
  `app.py` (FastAPI, `/predict`, `/health`, `/metadata`, input schema from the dataset
  modality — JSON tensor / tabular record / image upload / text), the exact
  preprocessing used in training (fitted stats saved by the training templates),
  model artifact, `requirements.txt`, `Dockerfile`, `README.md`.
- Formats: ONNX (+ dynamic int8 quantization), TorchScript, CoreML (coremltools,
  optional), TFLite (Keras path, optional), sklearn joblib / ONNX (skl2onnx optional).
- **Model registry** `~/.aime/registry/`: promote run → versioned model
  (name, version, metrics, source run, artifacts, stage: staging/production).
- Deploy dialog + Models dock; CLI `aime deploy`, `aime serve-model`.

## Phase 3 — Import & round-trip
- `core/importers/torch_fx.py`: `torch.fx` trace of an `nn.Module` → Graph; reverse
  map module class + attrs → block type + params (built from block definitions'
  `torch_layer` metadata), functional ops (add/cat/flatten/relu…) → merge/tensor
  blocks; unmappable modules → reported list (+ `layers.custom_module` block keeping
  source so export still works).
- `core/importers/onnx.py` (op → block), `core/importers/keras.py` (layer configs).
- Import from a `.py` file (module + class name), `.onnx`, `.keras`.
- Round-trip test: every sample + every block → generate → exec → import →
  regenerate → same output shape and parameter count.
- Custom block library: versioned user blocks, export/import `.aimeblocks` packs.

## Phase 4 — Data workspace
- `core/data/profile.py` (moved out of `ui/features/data_preview.py`): column types,
  missing, cardinality, numeric stats, histograms, correlations, target leakage /
  constant / ID-like column warnings, class balance; image-folder profile (sizes,
  channels, duplicates via `core/dataset_health.py`); text length stats.
- Data dock: paged table viewer (pandas QAbstractTableModel), image grid by class,
  profile charts, split preview (counts per split/class), **augmentation preview**
  (applies the graph's preprocessing chain to sample images in a worker subprocess).
- Data fingerprint (content hash) stored in run records; profile findings feed the
  Problems panel as lints.

## Phase 5 — Web version
- `ai_made_easy/server/` FastAPI app over `core/api.py`: REST for blocks / projects /
  validate / generate / summary / runs / sweeps / deploy / import / data; **WebSocket**
  run event streaming; project storage on disk; static frontend served at `/`.
  `aime web` command. Structured for auth later (dependency hook, no users yet).
- `web/` frontend: React + TypeScript + Vite + **React Flow** canvas; block library
  (search, categories) and inspector generated from block JSON schemas; live
  validation (debounced `/validate`), shape labels on edges, problems panel, code
  preview (Monaco-less highlighted pre), training charts, runs + sweeps pages, deploy
  page, data profile page; dark/light theme matching the desktop tokens.
- Built bundle committed under `ai_made_easy/server/static/` (so pip installs work
  without Node); Dockerfile for the web server.

## Phase 6 — Release 2.0.0
- Schema migration if the project format changes; README / ARCHITECTURE / CHANGELOG,
  screenshots of desktop + web; CI adds web build + API tests (+ Playwright smoke);
  version 2.0.0; PyInstaller still builds.

---

## Verification (each phase)
- `ruff check` + full `pytest` green locally and on CI (Ubuntu/macOS, 3.11/3.12).
- Phase 1: sweep of 4 trials on synthetic data completes; best params applied validate.
- Phase 2: serving package for a trained run starts with uvicorn in a subprocess and
  `/predict` returns the same prediction as the checkpoint in-process; ONNX output ≈
  torch output.
- Phase 3: round-trip test over all blocks + samples.
- Phase 4: profile tests on crafted CSV / image folders; UI placement tests.
- Phase 5: FastAPI TestClient tests per endpoint + WebSocket run stream; frontend
  `npm run build` + Playwright smoke (load, drag block, see validation).
