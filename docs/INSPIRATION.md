# Feature mining from open-source visual builders

We surveyed the open-source landscape of block/node-based ML tools, cloned
the three most relevant projects into `reference/`, and ported their best
UX ideas. This file records what exists out there, what we took, and what
we deliberately deferred.

## The landscape

| Project | Stack | What it is | Closest to us? |
|---|---|---|---|
| [Orange3](https://github.com/biolab/orange3) | Python + Qt | Visual data-mining/ML widget canvas, 20+ years old | Yes — same toolkit family, ML focus |
| [Langflow](https://github.com/langflow-ai/langflow) | React + FastAPI | Visual LLM/RAG flow builder with playground | Yes — our LLM side, exactly |
| [Ryven](https://github.com/leon-thomm/Ryven) | Python + Qt | General node-editor framework | Yes — pure canvas UX reference |
| Flowise | TypeScript | Visual LLM builder (Langflow alternative) | Similar to Langflow; not cloned |
| Node-RED | JavaScript | Generic flow programming (IoT) | Patterns only (palette/search/debug) |
| KNIME | Java | Enterprise analytics platform (open-source) | Giant; concepts overlap |

## Adopted (this session)

| Feature | Inspired by | Where |
|---|---|---|
| Sample Projects gallery (File menu, one-click editable starter graphs) | Orange "Examples" menu, Langflow template gallery | `main_window.action_open_sample` |
| Canvas PNG export (whole scene render, share your graph as an image) | Ryven `get_whole_scene_img` | `canvas.export_canvas_png` |
| Undo/Redo menu + Edit menu with discoverable in-canvas search | Ryven command stack + node chooser; OdenGraphQt already had Tab-search | `_build_menus_edit` |
| ⚡ Test Run — forward-pass smoke test of the current graph in a subprocess | Langflow playground ("run the thing you built") | `main_window.action_test_run` |
| Category colors on node headers | Orange category BACKGROUND colors | had it since Phase 1 — validated as the right call |
| Schema-driven blocks with typed params → auto UI | Langflow `inputs.py` field schema | had it (`ParamSpec`) — same idea, earlier |
| Analytic per-node summaries (shapes/params on canvas panel) | Orange StateInfo input/output summaries | Model Summary dock (Phase 4) |

## Adopted in the restructure pass (iteration 2)

| Feature | Inspired by | Where |
|---|---|---|
| Search-first palette (box + ranked results, ⏎ places at viewport center) | Orange quick-menu, Langflow sidebar search, Ryven node chooser | `panels/palette_search.py`, `registry.search()` |
| Header bar: app identity + editable project name left, actions right | Langflow flow bar | `_build_toolbar` |
| Floating canvas controls cluster (zoom ±, fit, snapshot) | Langflow CanvasControls, Ryven zoom widget | `panels/canvas_controls.py` |
| Inspector-on-selection (Properties raises when a node is selected, Summary otherwise) | Langflow inspection panel | `_wire_live_validation` |
| Per-node run status: Trainer node shows `epoch 3/10` live during training | Orange progress-on-node, Langflow build status | `_on_epoch_update` |
| Notes / groups on canvas (Backdrop node, "Notes" palette tab) | Langflow NoteNode, Orange annotations | palette relabeling |
| Live-switchable theme hub (Dark + Light, View menu) | Ryven FlowTheme/Design | `ui/theme.py` |

## Deferred ( with rationale — candidates for later)

- **Per-node status icons + progress bars during training** (Orange
  `messages.py` + `ProgressBarMixin`): the right evolution of our red-port
  validation once training UX deepens (show epoch/spin on the Trainer node).
- **Frozen/cached nodes** (Langflow `vertex.frozen`): needs a
  reactive-execution engine; our model is codegen-first.
- **Reactive layered executor** (Langflow `RunnableVerticesManager`): same
  reason — revisit if blocks ever execute live in-app.
- **Widgets embedded inside nodes** (Ryven per-port widgets): properties
  bin covers editing today; in-node widgets add canvas clutter at 123 blocks.
- **Live-switchable theme system** (Ryven FlowTheme painter hooks): we ship
  one polished dark theme; themes are polish, not function.
- **Export with secret scrubbing + version snapshots** (Langflow): matters
  once graphs contain credentials (assistant keys never enter graphs, so low
  pressure today).
- **Link/checkpoint reroute nodes** (Ryven special nodes): nice for dense
  graphs; our merge blocks keep layouts readable for now.
- **Sticky notes on canvas** (Langflow NoteNode / Orange annotations):
  OdenGraphQt BackdropNode partially covers grouping; text notes are a small
  future win.

The cloned sources stay in `reference/` (shallow, for reading only — not
part of the app or its tests).


## Full-catalog + safeguard audit (2026-09, second pass)

Re-cloned all three repos and re-audited against our then-123 blocks and
the new guardrail system.

**Verdict:** Ryven had nothing to port (no type checking; its error
indicator is unwired in the current snapshot). We filled every genuine gap
from the other two:

| Ported | From | As |
|---|---|---|
| Text splitter (chunk size/overlap/separator) | Langflow | `llm.text_splitter` + paragraph-aware RAG chunker |
| Document loader (glob + encoding) | Langflow | `llm.doc_loader` → RAG DOCS_GLOB |
| Chat memory (store/retrieve, n turns) | Langflow | `llm.chat_memory` → interactive chat loop in generated scripts |
| Structured output (JSON schema) | Langflow | `llm.output_parser` → schema instruction + json.loads + strict mode |
| K-Fold cross-validation (stratified) | Orange3 Test-and-Score | `train.kfold` → per-fold retrain + mean ± std |
| ROC-AUC | Orange3 ROC Analysis | `eval.roc_auc` → rank-based Mann-Whitney AUC (no sklearn needed) |
| Impute missing values | Orange3 Impute | `prep.impute` (mean/median/mode/constant/drop) both frameworks |
| Prompt `{var}` validation | Langflow | stray-brace + missing `{input}` checks on the Prompt Template |
| Validate-without-executing | Langflow | compile-only Python check on the Lambda block |
| Severity-icon tooltips | Orange3 | node tooltips now carry that block's ✖/⚠ issues (+ output shape) |
| LLM completeness | — | warning when LLM blocks exist without an HF Model block |

**Deliberately not ported** (revisited, still out of scope): classic-ML
models (SVM/RF/trees — sklearn widgets with no torch/keras codegen path),
visualization widgets (matplotlib views, not script blocks), web-search /
agent tools (network dependencies in generated scripts), feature-rank /
hyperparameter-sweep blocks (deferred), Paint Data (needs a drawing UI),
Orange's drag-time port highlighting and Langflow's impossible-wire
physics (we keep the smart-mix guard: undo + explain, so kids learn why).

## Research appendix — deferred with rationale (2026-09)

Evidence-backed features we deliberately did NOT build yet, so future
waves have the reasoning on file:

- **Whisper-style voice input** — local ASR pulls a ~150 MB model and a
  first-run download; conflicts with the fully-offline promise for
  classrooms. Revisit when a vendored tiny model (≤30 MB) meets quality.
- **Pose / hand-tracking projects** — needs MediaPipe-class dependencies
  (large, camera-permission-heavy, fast-moving APIs). The webcam capture
  + image-folder pipeline already covers "teach with your body" via
  photos.
- **Clustering / unsupervised mode** — no label-free objective in the
  block graph (loss/trainer assume targets); would need a parallel
  validation + mission arc. Evidence-wise, kids' first exposure is
  classification (AI4K12 grade bands), so supervised stays the spine.
- **Scratch-bridge export** — a JSON message protocol is easy, but the
  Scratch side needs a custom extension + network permission in the
  editor; the single-file HTML web demo already gives a shareable,
  offline "use it anywhere" artifact.
- **Per-neuron playground view** — TensorFlow Playground-style unit
  visualization conflicts with the block mental model (our neurons are
  layer-level). The Grad-CAM + feature-grid inspect view carries the
  "see inside" evidence goal for our age bands.
- **Local LLM copilot in-app** — transformers is already an optional
  extra for the LLM suite, but a chat copilot that edits graphs needs
  strong guardrails + a model download; missions + PRIMM cover guided
  help offline for now.
- **Hosted sharing** — anything hosted breaks the no-accounts,
  no-network promise for young learners; .aime bundles + the HTML demo
  are the sharing story (AirDrop/email instead of a server).

i18n groundwork started (title + chrome strings marked `tr()`); a full
string pass is queued with the localization study in mind.
