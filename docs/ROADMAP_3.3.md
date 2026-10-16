# AI Made Easy 3.3 — plan and status

*Status: shipped in 3.3.0. Every phase below is done; the items under "Moved to 3.4" are
the only parts of the plan that are not in this release, with the reason for each.*

## Context
3.2 trained anywhere, exported everywhere and connected the tools teams already use.
3.3 asked for four things: **more blocks, more features, a more polished UI, bug fixes and
more conveniences.** Three read-only surveys before the work started found:
- **Blocks:** 554 blocks covered the common layers. The gaps were modern transformer
  parts, embedding (metric-learning) losses, newer optimizers and explainability.
- **UI:** the web app lacked the desktop's command palette, shortcuts sheet, find and
  recent projects. Neither app had copy / paste, canvas notes, groups or drag-and-drop
  data. Native `confirm()` / `prompt()` and hard-coded colours remained, and several files
  were very large.
- **Bugs:** 14 confirmed, the worst being a download-token check that only worked on
  the client.

The process was the same as 3.2: one branch and one stacked pull request per phase, CI
green, heavy libraries behind extras, tiny synthetic test data, and no credentials
handled by the app.

## Step 0 — the download-token fix (in 3.2's last PR)
The server accepts `?token=` on GET requests, so download links work with
`AIME_WEB_TOKEN` set. **Done.**

## Phase 0 — bugs and hardening (#29)
- **Windows:** the run liveness check no longer sends a signal, which had terminated the
  process. A reused process id no longer counts as the run.
- **Web server:**
  - Runs a crash left "running" are failed at startup, and live views end.
  - Uploads have a size limit and are streamed to disk; results zips are checked
    before they are extracted.
  - The access token is hidden in the access log.
- **Run manager:**
  - Stopping a just-finished run doesn't mark it stopped, and following a remote run
    from two places starts one poller.
  - Event and log memory is bounded.
- **Remote and packaging:**
  - A failed Modal stop is reported.
  - The configured Python is used everywhere, not the app binary.
  - Docker and pip calls have timeouts.
- Temporary folders and files are cleaned up and unique.
- Ensembles whose size can't be estimated get a warning.

**Done**, with one test per fix.

## Phase 1 — more blocks (#30 and #33)
- **Transformer and LLM parts:**
  - rotary embedding, grouped-query attention, SwiGLU / GeGLU, a Llama block
  - Mixture of Experts, LayerScale, DropPath, FiLM, LoRA Dense
- **Vision and speech:** ViT patch embedding, ConvNeXt, Swin and Conformer blocks.
- **Losses:** ArcFace, CosFace, batch-hard Triplet and Supervised Contrastive
  (InfoNCE / NT-Xent with labels).
- **Optimizers and weight averaging:**
  - Lion, LAMB, Lookahead, SAM, Schedule-Free AdamW
  - EMA and SWA on the Trainer
- **Explainability:** permutation importance and Integrated Gradients, with the result
  shown with the run.
- **Classic ML:** Isomap, Nystroem, UMAP, Target Encoding and SMOTE (new `umap` and
  `imbalance` extras).
- **Recipes:** Tiny Llama, Vision Transformer, ConvNeXt and ArcFace embeddings.

**Done:** 583 blocks and 66 recipes. Each block is checked by the codegen harness or
trains in a test.

## Phase 2 — features and conveniences (#31)
- Copy, paste and duplicate blocks with their wires, between designs and between the two
  apps.
- Sticky notes, and named groups.
- Drop a data file on the window to start the wizard on it.
- Show a run's feature importance.
- Compare a run's design with the canvas.
- `aime new --data`.

**Done.**

## Phase 3 — UI refinement (#32)
- **Web:**
  - command palette, find, shortcuts sheet
  - a File menu with recent projects, and PNG export
  - in-app confirm and prompt dialogs, and a toast queue
  - loading, error and empty states
  - a fixed Ctrl/⌘ S, and a restore banner for autosaved work
  - keyboard and screen-reader access
- **Desktop:** a minimap and accessible names; `ui/context.py` split into mixins.
- **Both:** a three-step first-run tour, and one chart palette.
- **Deviation:** the web action list is written by hand (`web/src/actions.ts`) instead of
  generated from the desktop catalog. A test keeps the two in agreement on ids and
  shortcuts. This is simpler than a generator, and catches the same drift.

**Done.**

## Phase 4 — release 3.3.0
- The version is 3.3.0 everywhere.
- CHANGELOG, README, WEB.md and this document.
- FAMILIES.md and DEPLOY.md regenerated.
- New screenshots of the palette and the canvas.
- The Docker image and CI install the new extras.

The tag and release follow your confirmation.

## Moved to 3.4
- **SHAP.** It would add a heavy optional dependency (`shap`) for something Integrated
  Gradients and permutation importance already show for the supported inputs. Worth
  adding with image attributions (saliency for CNNs), where SHAP's kernels help most.
- **A SimCLR recipe.** Self-supervised contrastive pretraining needs a data pipeline
  that yields two augmented views of each unlabeled example. The supervised contrastive
  loss in 3.3 covers the labeled case; the two-view loader is a 3.4 item.
- **Collapsing groups.** Groups frame, name, select and move their blocks; folding a
  group into a single card is left for 3.4.
- **Cosine embedding and margin ranking losses.** These need paired or ranked examples
  rather than class labels, so they belong with the two-view pipeline above.
