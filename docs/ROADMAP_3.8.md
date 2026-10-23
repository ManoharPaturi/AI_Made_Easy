# AI Made Easy 3.8 — plan and status

*Status: built in PRs #65–#69, ready for your review. Nothing is merged or tagged until
you say so.*

## Phase 0 — platforms (#65)
- Python 3.13 and 3.14 join the CI test matrix (Ubuntu). **Done.**
- pandas 3: the suite runs on pandas 3 (pandas 2 stays tested on Python 3.11). **Done**,
  no code changes needed.
- `httpx2` in the dev extra (Starlette's test client prefers it). **Done.**
- The Windows test job runs as two halves (`AIME_SHARD=1/2`, `2/2`; files are split by
  weight in `tests/conftest.py`). **Done.**
- A test worker that dies no longer leaves the run hanging at 99%
  (`--max-worker-restart=0`, with `PYTHONFAULTHANDLER` for a traceback). **Done.**
- The MCP stdio round-trip test on Windows: the client now runs in its own process
  (`tests/fakes/mcp_client.py`), which no longer crashes the test worker. **Not done:**
  on Windows the server answers `initialize` and `list_tools` but not the first tool
  call. The test runs there as an expected failure that dumps the server's stacks (fixed
  in 3.9, see Still open).
- Making the Windows jobs required checks is your call (a ruleset change); it is not
  part of this plan.

## Phase 1 — more blocks (#66)
15 blocks, 762 in all, and 2 recipes. **Done**:
- Forecasters: NLinear, iTransformer, TSMixer, TimesNet. TimesNet keeps a running
  spectrum to pick its periods, which export freezes; ExecuTorch int8 can't load it, so
  that one export is blocked with a clear reason.
- Tabular: TabM.
- Layers (PyTorch): multi-head latent attention, QK-norm attention, gated attention;
  Mamba-2, Gated DeltaNet, gated linear attention, Griffin (RG-LRU); MobileNetV4 UIB,
  StarNet star block.
- Optimizer: ADOPT.
- Recipes: iTransformer forecaster; Regression with uncertainty.

Changes to the plan:
- **Dropped:** TabPFN (its weights are gated behind a sign-in and can't be tested
  offline). The losses proposed for 3.5's leftovers and an interval-coverage metric
  already existed (`coverage_80`).

## Phase 2 — features (#67)
- **Find a learning rate** (`core/lr_find.py`): a stage run imports the design's training
  script, raises the learning rate exponentially each batch (`BATCH_HOOK`, new in the
  PyTorch template) and stops when the loss blows up. The suggestion is the steepest
  descent of the smoothed loss before its minimum. Web Inspector and desktop Properties.
  **Done.**
- **Train more** (`core/train_more.py`): continues a finished PyTorch run from its best
  weights as a new tracked run. **Done.**
- **Results by group** (`core/groups.py`): recovers the run's exact test rows (PyTorch
  split indices; the classic split's row index), scores each group with the run's own
  metrics and flags groups at least 10% worse with at least max(10, 2%) rows. **Done.**
- API, web routes, both UIs, and tests (core, server, Qt, Playwright).

## Phase 3 — upkeep (#68)
- `training/catalog.py` 2037 → 1180 lines (helper sources to `train_helpers.py`);
  `torch_template.py` 1284 → 813 (`torch_template_data.py`, `torch_template_main.py`);
  `ui/features/experiments.py` 1145 → 842 (`experiment_dialogs.py`). Generated scripts
  are byte-identical (the golden tests pass unchanged). **Done.**
- The "still open" notes in ROADMAP_3.3–3.7 now say which release shipped each item.
  **Done.**

## Phase 4 — release 3.8.0 (#69)
- CHANGELOG, README, this roadmap, a screenshot of the learning-rate finder, version
  3.8.0. The tag and release follow your confirmation.

## Still open
Nothing: 3.9 fixed both ([ROADMAP_3.9.md](ROADMAP_3.9.md)).
- The MCP server on Windows stopped answering at its first tool call: the stacks it
  recorded showed numpy's extension loading while a thread blocked reading stdin. 3.9
  loads the registry before serving and moves the pipe off standard input.
- Core ML export of the new layers was untested: 3.9 runs it on macOS in CI and checks
  the converted programs against PyTorch.
