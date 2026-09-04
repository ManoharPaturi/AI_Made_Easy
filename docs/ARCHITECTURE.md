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
