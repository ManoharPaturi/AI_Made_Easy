# AI Made Easy 3.1 — plan

## Context
3.0 shipped 8 families, 29 tasks and 529 blocks. Every family is validated at design
time, generates runnable code, trains in the app, deploys, and works on desktop and web.
Each 3.0 phase listed what it put off ([ROADMAP_3.0.md](ROADMAP_3.0.md), "Not in this
phase"). 3.1 finishes the items that matter most to users. It does not open new model
areas.

Gaps that most limit users today, in order:
1. **Your own data stops at the demo for some tasks.** Generative, density, probabilistic
   and state-space recipes train only on their demo datasets.
2. **Pipelines stop at the basics.** There is no quantization-aware training, no
   per-layer learning rates and no stacking. Teachers must be PyTorch.
3. **Pretrained models are missing from the wizard.** Text and audio recipes train from
   scratch, even though the Hugging Face encoder blocks exist.
4. **Classic ML designs have no task.** `task_of` returns None for a random forest, so
   the wizard, explanations and project list can't say what it does.
5. **The block count is 529**, against 3.0's goal of 750+.

**Process (unchanged from 3.0):**
- One branch and one PR per phase, with CI green.
- Full `pytest` + `ruff` + web build / Playwright before merging.
- Heavy libraries go behind extras.
- Tests use tiny synthetic data. Real weights are tested only in the nightly
  `AIME_ONLINE` job.

---

## Phase 1 — Tasks for every family
- **Classic ML tasks:** classic designs resolve to tasks (multiclass, binary,
  regression, clustering, anomaly detection) from their estimator and data. Together
  with the classic recipes, this gives the wizard, Explain and the project list a task
  for every trainable family.
- **Project list:** the Open dialog's family / task chips then cover every family, and
  the desktop gets the same in **File ▸ Open Recent**.
- **Rules:**
  - a classifier on a continuous target
  - a regressor on a categorical target
  - clustering with a target column (ignored)
  - an anomaly detector with labels (used only for evaluation)
- **Status:** done. Classic designs resolve to multiclass / binary / regression and the
  new `clustering` and `anomaly_detection` tasks; LLM workflows (text generation,
  fine-tuning, RAG) and pipelines (their Train stage's task) have tasks too, so every
  sample but the architecture-only `skip_connection_mlp` has one. New recipes: k-means,
  Gaussian mixture, HDBSCAN, isolation forest, local outlier factor and three LLM
  starters (34 tasks, 51 recipes). Labels are optional for clustering / anomaly
  detection and only evaluate them; AutoML searches clusterings by silhouette.

## Phase 2 — Your data everywhere
- **Data readers** for the families that train only on demo data today:
  - image folders for VAE / GAN / diffusion
  - tables for density estimation (flows)
  - tables with group columns for PyMC
  - series for the state-space models
- **Detection and the wizard:** data detection learns these formats, and the wizard
  offers the matching recipes with the user's data swapped in.
- **More data sources:**
  - user / item feature tables beside interaction CSVs (recommenders)
  - graph-level CSV datasets (one graph per id)
- **Tests:** each new reader gets a tiny synthetic fixture and an end-to-end training
  test.
- **Status:** done. Image folders (flat or per class, colour or grey) train the VAE, DCGAN
  and diffusion recipes. Tables train the flows and, with a group column, the hierarchical
  PyMC models (priors scaled to the data). Series train the structural, SARIMAX and HMM
  recipes, with the season length taken from the timestamps. Interaction logs pick up
  user / item feature tables, and the new Graphs CSV block reads one graph per id.

## Phase 3 — Pretrained recipes
- **Recipes on the existing encoder blocks:**
  - text classification on a Hugging Face encoder
  - audio classification on wav2vec 2.0 / HuBERT
  - image classification on timm / DINOv2, with linear-probe and fine-tune variants
- **Wizard behaviour:**
  - shows download size and licence (from `core/zoo`)
  - offers pretrained recipes only when the budget fits
- **Tests:** offline tests run with `weights="none"`. The nightly job runs one real
  fine-tune per modality.
- **Status:** done. Eight recipes: a linear probe and a fine-tune each for a Hugging Face
  text encoder, a wav2vec 2.0 / HuBERT / WavLM speech encoder, timm image models and
  DINOv2 (59 recipes). Recipe cards list each checkpoint's download size and licence
  (new `hf_encoders` zoo catalog). With a budget, pretrained recipes that do not fit are
  left out. The text encoder gained `weights = none` and cost estimates, and frozen
  encoders no longer count as trainable in budgets.

## Phase 4 — Advanced pipelines
- **Fine-tune:** discriminative learning rates (per layer group, geometric decay) next to
  the freeze / unfreeze schedule.
- **Quantization-aware training stage:** prepare → fine-tune → convert, compared with
  post-training dynamic int8 in the stage metrics.
- **Stacking ensembles:** a meta-learner (logistic / ridge) trained on out-of-fold
  predictions from a Cross-validate stage.
- **Classic teachers:** distill from scikit-learn / boosting teachers (soft labels from
  `predict_proba`).
- **Composite blocks:** show an embedded design inline on the canvas (expand / collapse),
  on desktop and web.
- **Status:** done. Fine-tune gets `lr_decay` and `layer_groups`. The new QAT stage
  reports float, dynamic int8 and QAT int8 metrics; on digits, QAT int8 scored 96.6%
  against 96.1% float, at 0.05 MB against 0.20 MB. Cross-validate saves out-of-fold
  predictions and can refit; Ensemble `stack` matched the best member (96.6%) and beat
  the plain average (95.5%). A random-forest teacher distilled into an MLP reached 92.7%.
  Stages expand on the canvas to show their design. New sample:
  `pipeline_advanced.json`.

## Phase 5 — Block coverage
Prioritised by what users reach for. Each block gets a runtime shape test and a
FAMILIES.md entry automatically.
- **Vision:**
  - FPN neck block
  - YOLO-style detector composed from blocks
  - GIoU / DIoU box losses
  - Mask2Former (Hugging Face)
- **Sequences:**
  - TFT-lite
  - AST audio backbone
  - LSTM projection variant
  - BPE tokenizer block for language models
- **Probabilistic:**
  - MvNormal / LKJ distributions with plate shapes
  - MMHC structure learning
  - conformalized quantile regression
  - time-varying coefficients in state-space models
- **Generative:**
  - `generative` extra (diffusers) for LoRA fine-tuning of Stable Diffusion
  - Inception Score
- **RL:** CNN policies on image observations, plus Box2D environments.
- **Coverage goal:** a contributor guide section on adding a block, and the
  `verify_codegen` harness reporting coverage per family.
- **Status:** done (544 blocks). The YOLO-style detector from blocks reached mAP@.5 0.90
  on the shapes in three epochs; BPE took the tiny GPT from 1.42 to 0.81 bits per byte;
  LKJ recovered a negative intercept–slope correlation; CQR calibrated 92% raw intervals
  to 94% coverage at α = 0.1; time-varying coefficients followed a drifting price effect
  (−2 → −5.4) and cut the forecast error by 22–42%; a CNN policy learns PixelChase from
  pixels while an MLP on the same pixels does not. Stable Diffusion LoRA stays in scope
  behind the `generative` extra: offline tests use a tiny random model, and the real
  checkpoints download only when a design asks for them. The codegen harness runs all
  198 layer blocks (`--coverage`); `CONTRIBUTING.md` explains how to add one.

## Phase 6 — Release 3.1.0
- **AutoML:** a multi-objective mode (accuracy vs. latency Pareto front) in AutoML and
  the leaderboard.
- **Docs:**
  - new desktop screenshots
  - FAMILIES.md regenerated
  - CHANGELOG
- **Version:** 3.1.0, plus nightly and CI green.

---

## Verification (every phase)
- `ruff` + full `pytest` + `npm run build` + Playwright, locally and on CI.
- **Per new block:** generated code executes and gives the designed output shape.
- **Per new reader / recipe:** a tiny synthetic end-to-end training run and a web smoke
  step.
- **Phase 1:** every sample and every recipe's design has a task.
- **Phase 4:** QAT keeps accuracy within 1 point of float on the digits pipeline, and
  stacking beats the best single member or matches it within noise.

## Open questions for review
- Is the order right? Phase 2 (your data everywhere) could come first if real datasets
  matter more than classic tasks.
- How hard is the 750-block target? Phase 5 names about 20 valuable blocks, which would
  bring the total to about 550. Reaching 750 would mean padding with low-value variants,
  so a coverage goal per family may be the better target.
- Should Stable Diffusion LoRA (multi-GB downloads, GPU only) be in scope, or stay
  deferred?
