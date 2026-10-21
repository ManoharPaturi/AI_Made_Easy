# AI Made Easy 3.5 — plan and status

*Status: released as 3.5.0. You approved the plan below; every phase is merged.*

## Phase 0 — faster test runs and repo tidying (#49)
- A plain `pytest` runs on four workers (`-n auto --maxprocesses=4 --dist loadfile`: one
  test file per worker, so module fixtures still train once) and leaves out the 308
  tests in `tests/slow_tests.txt` (10 s or more). `AIME_TESTS=all` or `--run-slow` runs
  them; a slow test named on the command line always runs; CI sets `AIME_TESTS=all` in
  every job, so it still runs everything. **Done:** 3 min 49 s on an 8-core Mac against
  45–60 min serially. `scripts/find_slow_tests.py` rewrites the list from a timed run.
- `test_tcn_receptive_field` seeded its random numbers after building the network, so its
  weights depended on the tests before it; it seeds first now.
- `ai_made_easy.egg-info/` is no longer tracked. **Done.**

## Phase 1 — the web canvas catches up on folded groups (#46)
- A folded card is draggable; its hidden blocks move by the same amount, so saving,
  reopening, undo and redo keep the move. Wires that end inside the card are dashed
  stand-ins. **Done**, with a Playwright test.

## Phase 2 — more explainability (#47)
- **Attention Maps** (`eval.attention_maps`). While predicting a few test samples,
  scaled dot-product attention runs as softmax(QKᵀ)V so its weights can be kept; this
  covers `nn.MultiheadAttention`, the Transformer layers (fused fast path off
  meanwhile) and blocks that call `F.scaled_dot_product_attention` (GQA, causal masks).
  A test checks the recorder gives the same output as PyTorch's attention. **Done.**
- **Audio Saliency** (`eval.audio_saliency`): gradient saliency over each test clip's
  spectrogram, or under a raw waveform. **Done.**
- Both apps show Grad-CAM, attention and saliency pictures in the run's gallery.

## Phase 3 — try a trained model (#48)
- A finished run's predictor loads once in a worker process (the run's folder and
  interpreter) and answers with the run's own `infer()`; the two most recent stay
  loaded and close with the app. Forms in both apps follow the input kind. Nothing is
  sent anywhere and no package or server is built. **Done.**
- **Deviation:** the plan mentioned Grad-CAM inside the Try it result for images. It
  isn't there: Grad-CAM needs the trained network itself, and the predictor only
  exposes `infer()`. Grad-CAM stays in the run's gallery (`eval.grad_cam`).

## Blocks — more from the libraries you use (#50)
You asked, with the plan approved, for the blocks of the well-known libraries. A scan of
`torch.nn`, `torch.optim`, `keras.layers` / `losses` / `optimizers` and scikit-learn
against the catalog found the gaps; this adds the ones that fit a design: 21 layers,
6 losses, the Muon optimizer, 35 estimators and 21 transformers. **Done.**
- **Left out on purpose:** lazy layers (the designer infers shapes), multi-GPU wrappers,
  deprecated classes, RNN cells, containers, Keras preprocessing layers the tabular
  pipeline already covers, TimeDistributed (Dense already applies per step), and
  scikit-learn meta-estimators that need other estimators passed in (stacking, voting,
  one-vs-rest, RFE, SelectFromModel), and the non-pipeline manifold methods (t-SNE, MDS,
  spectral embedding have no `transform`).
- **Still open:** Gaussian NLL (needs a mean and a variance output per target),
  multi-label margin and the explicit triplet losses (need different target formats),
  L-BFGS (needs a closure in the trainer).

## Phase 4 — release 3.5.0
- CHANGELOG, README, FAMILIES / DEPLOY regenerated, a screenshot of Try it, this
  roadmap, version 3.5.0. **Done.**

## Still open
- The items listed under Blocks above.
- Hyperparameter sweeps in the web app (proposed, not in this plan).
