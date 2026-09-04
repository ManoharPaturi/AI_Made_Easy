# UI Architecture

Rebuilt from the structural logic of Orange3, Ryven, and Langflow (see
`docs/INSPIRATION.md` for the feature-mining pass that preceded this).
The rules below are **enforced by tests**, not aspirational.

## Structure

```
ui/
├── app.py                # entrypoint: env workaround, QApplication, theme, context+workbench
├── context.py            # AppContext — composition root; the ONLY place signals are wired;
│                         #   thin act_* intent slots driven by the actions catalog
├── workbench.py          # Workbench(QMainWindow) — SHELL ONLY: setup_actions → setup_ui →
│                         #   setup_menus phases, docks, QSettings save/restore, close guard
├── actions_catalog.py    # actions as DATA: ActionSpec list -> QActions/menus/ShortcutsDialog
├── stores.py             # ProjectStore, RunStore, ValidationStore, LogBus — single-concern,
│                         #   granular signals; components subscribe to slices
├── theme.py              # ThemeService + THEMES (no module globals)
├── services/             # functionality objects (no widgets, no OdenGraphQt)
│   ├── graph_service.py      # ONE settle pipeline (debounce -> graph_settled), validation,
│   │                         # block placement, expansion, node-status renames
│   ├── process_service.py    # THE subprocess pattern: train / test-run / runtime exports
│   ├── export_service.py     # THE codegen dispatch (render targets) used by files + preview
│   └── project_service.py    # project IO, samples, dirty tracking via ProjectStore
├── canvas/               # the ONLY package that imports OdenGraphQt  [enforced]
│   ├── area.py               # CanvasArea page; factories for palette/properties widgets
│   ├── adapter.py            # CanvasController: OdenGraphQt <-> core IR adapter
│   ├── node_factory.py       # node classes + bidirectional type maps (O(1) both ways)
│   ├── painter.py            # flat block renderer (NodeItem.paint patch) — solid family
│   │                         #   fills, bold ink centered labels, no chrome; proxy mode off
│   └── templates.py          # user-template filesystem store (~/.aime/templates)
├── features/             # one module per page region; dumb components (render + emit intents)
│   ├── header.py             # identity + name field two-way bound to ProjectStore
│   ├── palette.py            # search-first palette + pretty tabs + Notes tab
│   ├── inspector.py          # InspectorStack (switching policy) + Summary/Preview/Assistant
│   ├── runconsole.py         # RunConsoleStack (auto-raise policy) + Console/Training pages
│   ├── workspace.py          # the ONE page: card layout (blocks | canvas+runs | inspector)
│   │                          #   with locked splitters — panels can resize, never float
│   └── canvas_controls.py    # floating zoom/fit/snapshot cluster
└── dialogs.py            # BaseDialog + SampleGallery + SaveTemplate + Shortcuts (from catalog)
```

## The rules (and their tests)

| Rule | Origin | Enforced by |
|---|---|---|
| Shell builds in phases actions→UI→menus; menus only assemble existing actions | Orange | `test_workbench_is_layout_only` (≤15 methods, no business names) |
| OdenGraphQt lives only in `ui/canvas/` — the adapter boundary is real | Ryven | `test_odengraphqt_only_imported_inside_canvas_package` (AST lint) |
| Core stays Qt-free | Ryven | `test_core_stays_qt_free` (AST lint) |
| Actions are data; every spec resolves a handler | Langflow | `test_every_catalog_slot_resolves_on_context` |
| Single-concern stores with granular signals; one write path per state | Langflow | `test_project_store_name_roundtrip_no_stale_state`, `test_run_store_single_writer_state_machine` |
| One settle pipeline; one codegen dispatch; one subprocess pattern | — | `test_workbench_smoke` end-to-end |
| One colour per functional family (conv+transformer+arch = model gold; activations one colour; …) | reference design | `tests/test_color_system.py` (family purity + distinctness + ink contrast) |
| Geometry/state persisted symmetrically, versioned | Orange/Ryven | QSettings `aime/workbench` v1 |

## Guardrails — the constraint catalog

Every rule a learner can break, and where it's enforced. Core rules are
Qt-free and unit-tested (`tests/test_guardrails.py`); the canvas applies
the visuals (badges, red ports, tips); the context gates runs.

| # | Rule | Severity | Enforced where |
|---|------|----------|----------------|
| 1 | Every numeric param declares `minimum` (and `maximum` where natural) and stays inside it | error | `Graph.validate()` + spinbox ranges (registry-audit test) |
| 2 | Param type matches; enum value ∈ options | error | `Graph.validate()` |
| 3 | Cross-param relations (`BlockDefinition.checks_fn`, e.g. MHA `embed_dim % num_heads`) | error | `Graph.validate()` |
| 4 | No tensor↔config wires — undone instantly at wire time with a friendly message (smart-mix guard) | error | `adapter._on_port_connected` + `GraphService.guard_message` |
| 5 | Every tensor input connected | error | `Graph.validate()` (unchanged) |
| 6 | Shape failures carry `node_id` (`infer_shapes_detailed()`); `infer_shapes()` still raises for codegen | error | core/graph.py |
| 7 | One wire per input (duplicates suggest Concat/Merge) | error | `Graph.validate()` |
| 8 | Exactly one Input + one Output, Output reachable | error | `Graph.validate()` |
| 9 | No cycles | error | `Graph.validate()` (unchanged) |
| 10 | Off-path tensor blocks | warning | `Graph.validate()` |
| 11 | Training completeness: duplicate Trainer is an error; missing loss/optimizer or orphan scheduler warn | error/warning | `Graph.validate()` |
| 12 | Dataset features == Input shape volume | error | `Graph.validate()` |
| 13 | 💡 fix tips appended to known failures (`core/suggestions.py`: add Flatten, match channels, drag a wire…) | message | `validate()` post-process |
| 14 | Negative dims (`dim=-1`) keep negative spin ranges | — | blocks audit + `node_factory` |

Surfacing: ✖ red badge + red ports on error blocks, ⚠ amber badge on
warnings (painter), 🩺 Checks list at the top of 📋 Summary with
click-to-jump (`adapter.select_and_center`), shape tooltips on hover,
issues logged to the Console, and ▶ Train / ⚡ Test Run blocked by a
friendly checklist while errors exist (`AppContext._guard_run`).

The properties panel honors bounds for real: `PropSpinBox`/`PropDoubleSpinBox`
get working `set_min`/`set_max`, and float boxes grow decimals so tiny
values (lr 1e-4, eps 1e-8) survive editing (`ui/canvas/prop_widgets_patch.py`).

## Signal flows

```
canvas mutations ──► GraphService (400ms debounce) ──► graph_settled(Graph) ──┬► ValidationStore + adapter.apply_validation
                                                                             ├► PreviewPage (via ExportService.render)
                                                                             ├► SummaryPage (summarize)
                                                                             ├► AssistantPage.set_graph
                                                                             └► ProjectStore.mark_dirty

header/buttons ─► context.act_* ─► services ─► ProcessService.run_* ─► RunStore.state_changed ─┬► TrainingPage (buttons/status)
                                                                                                ├► HeaderBar.set_running
                                                                                                └► RunConsoleStack auto-raise
                                       epochs ─► TrainingPage.on_epoch + GraphService.set_node_status("train.trainer", "epoch n/m")

ProjectStore.name_changed ⇄ HeaderBar.name_edit (loop-guarded two-way binding — the stale-name bug class)
LogBus.logged ─► ConsolePage (the only renderer)
```

`AppContext._wire()` is the ONLY place signals are connected; everything a
button can do depends on it having run. `tests/test_wave1.py::
test_context_wiring_smoke` clicks/emit-probes every intent signal on a booted
context — a regression there once shipped a GUI where Train did nothing
while the methods beneath still worked.

## Wave 1 — See Inside (post-training insight)

After a training run the generated script dumps, into the run workspace:
`predictions.json` (first 300 test examples, per-class probs), `mistakes.json`
(first 50 misclassifications) and, when the dataset has no per-sample files
(e.g. torchvision MNIST), `samples/<idx>.png` thumbnails rendered from the
batch tensors themselves (`_save_sample_png`). Three insight buttons on
📈 Training enable once `predictions.json` exists:

- **🔍 Mistake Museum** (`features/mistake_museum.py`) — cards with thumbnail,
  actual vs guessed, top-3 confidence bars, and "what went wrong?" remedy
  chips with kid-worded advice.
- **👀 What is it looking at?** (`generate_inspect` + `features/inspect_view.py`)
  — regenerates the script in inspect mode and runs it as a subprocess:
  manual Grad-CAM (forward hook with `out.retain_grad()`) heatmap overlay on
  the input, a 2-D feature-map grid of the first conv layer, and a plain
  sentence ("The model guessed class 5 (18%)…"). Sample stepper + "my own
  image" re-run through the same launcher.
- **🪪 Report Card** (`core/model_card.py` + `features/report_card.py`) —
  auto-fills accuracy, dataset snapshot and worst confusions from the run
  artifacts into a five-field kid card (two fields kid-written), markdown
  preview, save as `.md`.

`CelebrationOverlay` paints confetti *before* the headline card so falling
emoji never obscure the message.

## Wave 2 — Teach with your world (perception)

- **📷 Capture dialog** (`features/capture.py`) — hold-to-record examples
  into an Image Folder: Qt Multimedia only (QCamera + QMediaCaptureSession
  + QVideoWidget + QImageCapture), frames saved mirrored, ~3 per second
  while held. Class picker is seeded from the root's subfolders (➕ makes
  new ones, names sanitised). A blocked/inactive camera (macOS privacy)
  is detected via `camera.isActive()` after 2.5 s and gets a plain-words
  hint instead of a blank viewfinder; the first open explains the one-time
  permission prompt. Entry point: double-click an Image Folder block →
  📷 button.
- **🎤 Sounds tab** — same dialog, optional `sounddevice` (pip extra
  `perception`): each hold records ≥0.4 s and becomes one log-spectrogram
  PNG (`torch.stft`, amber→red map) inside the class folder — the CNN
  pipeline trains on it unchanged.
- **🔴 Live prediction** (`features/live_predict.py`, Training page) —
  imports the run's generated script as a module (model class, INPUT_SHAPE,
  NORM_* are module-level; main() is guarded), loads the checkpoint in a
  QThread worker, and predicts camera frames at ~8 FPS with the same
  tensor math as the Image Folder loader (PIL resize → /255 → optional
  normalize). Frames are tapped through a session QVideoSink
  (`QCamera.videoSink()` is not bound in PySide6). Top guess + per-class
  bars; 📁 try-a-photo works without a camera. Worker/camera stop on
  finished/close (accept() skips closeEvent — cleanup hooks both).
- **🩺 Dataset health meter** (`core/dataset_health.py` pure rules +
  `_HealthMeter` in the data preview) — per-class balance bars (red when
  < 10 photos), empty-class / too-few / imbalance-ratio (≥3×) / exact
  duplicate (sha256) findings with 💡 kid hints; the same findings are
  appended to the 📋 Checks list as warnings by
  `context._dataset_health_issues` (fs scan in the UI layer, rules pure in
  core).

## Wave 3 — 🧭 Pedagogy

- **🔮 Predict-before-training** (`features/predict_gate.py`) — every valid
  ▶ Train first opens the guess dialog (😵/🙂/🤩; "just train, no guess"
  skips; QSettings `aime/pedagogy/predict_gate` can disable). A
  confidently-wrong guess (two bands off; bands at 0.60/0.85) triggers the
  😲 **Surprise** dialog after the run with pointers to the museum/health
  meter — POE + hypercorruption, evidence-backed.
- **PRIMM missions + dual tracks + quizzes** (`features/missions.py`) —
  every mission is 🔮 Predict → ▶ Run → 🔍 Investigate → 🔧 Modify → 🎨
  Make; stages complete on events the context forwards (`mission_event`:
  guess made, run finished, museum/inspect opened, graph changed between
  runs → modify, own project trained → make). Tracks 🟢 8–10 / 🔵 11–14;
  finishing all stages offers a 3-question checkpoint quiz; wrong answers
  point back at the mission.
- **⚖️ Bias arc** — `samples/bias_arc/{biased,fair}` (circles vs triangles
  on grass/sand, generated PNGs) + two sample graphs. The biased set is
  flagged live by `background_shortcut` (every class on its own
  background, ≥90% luminance-bucket purity per class) in the health
  meter; the trained biased model demonstrably classifies by background
  (a circle on sand is called a triangle), and the fair sample fixes it.
  Dataset file paths in generated scripts resolve to absolute at
  generation time (training runs in a temp workspace).
- **Anti-anthropomorphism + train/use modes** — copy swept ("the model
  got X right", never "smart/thinks"); after a run the statusbar shows
  🎁 Using mode and TrainingPage carries a 🔒 model-locked chip (cleared
  when retraining starts).

## Wave 4 — 🌐 Share & shine

- **Single-file web demo** (`core/web_export.py`, ⬇ PyTorch → Export web
  demo) — the trained torch model converts to tfjs-layers layers (Conv2d
  OIHW→HWIO, Linear transposed, BatchNorm running stats, pool/flatten/
  activation/dropout/global-pool; anything else fails with a kid-worded
  message) with weights as base64, inlined into one double-clickable
  ~1.5 MB HTML together with the vendored tf.min.js — drag-drop a photo
  (or click), resize → /255 → optional normalize, bars with class names.
  `?selftest=1` (or a `true` swap) builds and predicts on zeros and
  prints the result — live-verified in Chrome.
- **.aime bundle** (`core/bundle.py`, File menu) — zip with manifest
  (format/version/app version/created), graph.json, model card, canvas
  thumbnail, size-capped dataset/, run/ (checkpoint + train script +
  predictions). Opening validates the manifest and offers: load their
  project, or **🔁 swap-test** — their checkpoint on YOUR photos via a
  generated eval script (image-folder models), ending in a comparison
  card ("their model on YOUR pictures: 62%") — generalization made
  social.
- **Animated wires** (`ui/canvas/painter.py`) — while RunStore is
  RUNNING a single shared `WireFlowAnimator` timer crawls a teal dash
  overlay along every pipe (pipes self-register in a WeakSet; only they
  repaint).
- **⌨️ blocks ↔ python** — a ⌨️ button in the canvas controls reveals a
  code pane under the canvas (vertical splitter in the canvas card),
  live-synced to the preview renderer through `context.side_code`.
- Dataset paths the swap-test uses resolve to absolute at use time
  (subprocesses run in temp workdirs); `run_script` runs now finish the
  RunStore (FINISHED/FAILED) instead of dangling at RUNNING.

## What died in the rebuild

- The 701-line / 37-method / ~20-concern MainWindow god object.
- The 15-responsibility CanvasController (now a focused adapter; template IO,
  demo data, and the settle pipeline moved out).
- Three different subprocess patterns (now one ProcessService).
- The parallel codegen dispatch in the preview panel (now one table in
  ExportService; the preview page is dumb).
- Manual project-name syncing with its stale-state bugs (now one store).
- Double validation on canvas-replacing actions (single settle path with a
  load-guard).
