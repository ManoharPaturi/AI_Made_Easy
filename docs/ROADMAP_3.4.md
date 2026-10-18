# AI Made Easy 3.4 — plan and status

*Status: ready for 3.4.0. Phases 1–6 are done. The tag and release wait for your
confirmation; Phases 3–6 followed a draft plan you haven't approved yet.*

## Phase 1 — what 3.3 deferred (`feat/v34-phase1`)
- **SHAP values** (`eval.shap_values`, `explain` extra). Expected gradients
  (`shap.GradientExplainer`) against training rows. It writes the same `explain.json` as
  permutation importance and Integrated Gradients, and both apps show it with the run.
  **Done.**
- **Collapsing groups.** A group folds into one card and opens again where the card was
  moved. Its blocks, their parameters and the generated code don't change.
  `meta.groups[].collapsed` is saved with the design. **Done.**
  - **Web:** wires to hidden blocks are re-routed to the card.
  - **Desktop:** the card is a backdrop of at least 230 × 106, and it remembers its blocks
    because a backdrop otherwise finds them by what it covers. Wires to folded blocks are
    hidden.
- **SimCLR.** The `train.loss_nt_xent` loss and an image pipeline that yields two
  augmentations of each training and validation image. The loop folds each pair into one
  batch, and each view's positive is its twin. The test set uses single views. The
  `image_simclr` recipe uses a CNN encoder with a projection head. **Done.**

## Phase 2 — the rest of the deferred list (`feat/v34-phase2`)
- **Cosine Embedding and Margin Ranking losses** (`train.loss_cosine_embedding`,
  `train.loss_margin_ranking`). **Done.**
  - *Cosine embedding:* every same-class pair in a batch pulls its cosine towards 1, and
    every other pair pushes it below a margin.
  - *Margin ranking:* around each example, every same-class neighbour must be more
    similar than every other-class one, by the margin.
  - **Deviation:** the plan was a pairs-dataset block first. Instead, the pairs are made
    from the class labels in each batch, the way the batch-hard triplet loss already
    works. That way the losses work with every dataset that has classes, and they score
    accuracy the same way as the other metric losses. A dataset of explicit labelled
    pairs (e.g. two columns of ids and a same / different label) stays open, for when
    someone needs it.
- **Wires on folded desktop groups.** Wires that cross a folded group's border are drawn
  as dashed stand-ins from the card's edge to the block outside (or to another folded
  card), and they follow the card. Only the drawing changes: the real wires are kept, so
  the design and its code don't change. This replaces the proxy ports the plan had; the
  result looks the same and the canvas library isn't touched. **Done.**

## Phase 3 — explaining image models (`feat/v34-phase3`)
- **Grad-CAM Heatmaps** (`eval.grad_cam`). After training, the last convolution's
  Grad-CAM is drawn over the first test images: each picture shows the image, then the
  heatmap over it, and its file name gives the true and the predicted class. The pictures
  go to the run's `eval_samples/`, so the web shows them with the run's samples and the
  desktop shows them on the Experiments page. **Done.**
- Designs that aren't image classifiers skip it with a warning. Metric losses are skipped
  too, because their model outputs an embedding, not class scores.
- **Finding:** the desktop already had Grad-CAM in the Inspect dialog, for one image at a
  time. The new block reuses that function, so the two can't drift, and the generated
  script defines it once.

## Phase 4 — labelled pairs (`feat/v34-phase4`)
- **`data.pairs` (Labelled Pairs):** each row is two items (the `a_*` and `b_*` columns of
  a CSV, TSV or Parquet file) and a 0 / 1 label. With no path, it generates demo pairs.
  **Done.**
- **How it trains.**
  - The generated script keeps the model class as the encoder and wraps it under the same
    name, so everything that builds the model gets the pair version. For a pair
    `(batch, 2, …)` the wrapper returns both embeddings; for a single item it returns that
    item's embedding, which is what export and serving use.
  - The two pair losses read explicit pairs when they get them. Cosine Embedding gives
    `1 − cos` for same pairs and `max(0, cos − margin)` for different ones. Margin
    Ranking is `nn.MarginRankingLoss` on the two items' scores, so its model ends in one
    unit.
  - Each loss also turns a pair into one logit, so the task is binary and the binary
    metrics (accuracy, ROC-AUC, …) apply unchanged.
  - Other losses, and ranking models with more than one output, are rejected with a
    message.
- **Recipe:** *Pairs — same or different (siamese)*. **Done.**
- **Not included:** pairs of images (two image paths per row) and pairs in Keras, where
  the metric losses don't exist yet. The data wizard doesn't detect a pairs table; you
  add the block yourself.

## Phase 5 — bugs and polish (`feat/v34-phase5`)
A read-only audit of 3.3 and 3.4's newest code found four bugs; each fix has a test.
**Done.**
- **Serving metric-learning models (from 3.3).**
  - `infer()` passed an ArcFace / triplet / contrastive model's embedding to
    `describe()` as if it were class logits, so served labels and probabilities were
    wrong.
  - The trained loss's state (class centres or averages) is now saved with the inference
    state and rebuilt to score embeddings, and each result carries its embedding. Pair
    models serve each item's embedding (or its score, for ranking).
  - Runs trained before the fix have no saved scorer and say so; retrain them.
- **Undo of a folded group (desktop).** Undoing the group's creation removed the card but
  left its blocks hidden. The canvas now keeps the hidden blocks in step with the folded
  groups it has after every change, so undo shows them and redo folds them again.
- **A moved folded card (desktop).** Saving wrote the hidden blocks at their old
  positions and reopening fitted the card around them, so the move was lost. The blocks
  are now saved where the card will open them. The web's card can't be dragged, so it
  isn't affected.
- **Pairs and feature explanations.** Permutation importance, Integrated Gradients and
  SHAP would have treated a pair's two items as two features; they are skipped with a
  warning. Grad-CAM's sample count is capped at 12, the most a run's samples view shows.
- **The Qt "shared QObject was deleted directly" message. Not fixed.** Since Phase 2,
  full local runs (1799 and later 1814 tests) end with this one line, printed after
  the summary as the process exits. Every test passes and the exit code is 0. It doesn't
  appear when the nine desktop test files run on their own, together or one by one, so
  it depends on the whole run's teardown.
  - One guess, that `scene.changed` connected to plain Python callables leaves hidden
    receivers behind, was tested by making the stand-in manager a QObject child of the
    scene. The message stayed, so the change was dropped.
  - The 26 test files that use Qt, run in two groups (233 and 256 tests), don't print it
    either. It needs most of the suite in one process, so it's an interpreter-teardown
    effect, not one feature's.
  - It stays open.

## Phase 6 — release 3.4.0 (`feat/v34-phase6`)
- The version is 3.4.0 everywhere (the same files as 3.3), with the CHANGELOG, README
  and this document updated, and FAMILIES.md and DEPLOY.md regenerated. There's a new
  screenshot of a folded group on the desktop
  ([desktop_folded_group.png](images/desktop_folded_group.png)). No new extras: `shap` joined `explain` in
  Phase 1, and the Docker image and CI already install it.
- The tag and release follow your confirmation.

## Still open
- The next round of 3.4 features, to be agreed.
