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
│   └── …                 summary, fixes, suggestions, model card, bundles, runner
├── ui/                   PySide6 application
│   ├── context.py        composition root: builds and wires everything
│   ├── workbench.py      main window shell (layout only)
│   ├── actions_catalog.py  every action declared as data
│   ├── theme.py, icons.py  design system
│   ├── canvas/           the only package importing OdenGraphQt
│   ├── features/         panels and dialogs (presentation only)
│   └── services/         graph settle pipeline, processes, exports, projects
├── worker/               subprocess wrapper streaming JSON training events
├── mcp/                  MCP server over the core engine
└── cli.py                headless command line
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
