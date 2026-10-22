# AI Made Easy 3.6 — plan and status

*Status: built in PRs #52–#57, ready for your review. You approved the plan below, with
more blocks; nothing is merged or tagged until you say so.*

## Phase 0 — the blocks 3.5 left open (#52)
- **Gaussian Head** (`bayes.gaussian_head`): a mean and a variance per target, trained
  with the Gaussian negative log-likelihood (torch's `GaussianNLLLoss` with a learned
  variance). It is a one-component Mixture Density Head, so intervals, conformal
  prediction and serving work unchanged. **Done.**
- **Multi-Label Hinge** (`train.loss_multilabel_margin`): torch's `MultiLabelMarginLoss`
  on multi-hot targets, PyTorch and Keras. **Done.**
- **Labelled Triplets** (`data.triplets`): anchor / positive / negative tables for the
  Triplet loss. **Done.** (Labelled pairs already arrived in 3.4.)
- **L-BFGS** (`train.lbfgs`): the training loop steps with a closure. **Done.**

## Blocks — more from the libraries you use (#53)
11 layers, 3 optimizers, 2 schedules, 2 losses, 7 metrics, 4 augmentations and 8
scikit-learn models and transformers (see the CHANGELOG). **Done.**

## Phase 1 — Try it, part two (#54)
- **Predict a file** in both apps: a CSV of predictions back. **Done.**
- **Grad-CAM in Try it.** **Done, differently:** the plan was a `load_model()` in
  generated scripts; instead the Try it worker computes Grad-CAM from the network it has
  already loaded, so generated code is unchanged and older runs explain too.
- **Deviation:** the "ready-to-serve bundle" is left out: the deploy package already is
  one (FastAPI server, Dockerfile, verified exports).

## Phase 2 — comparing runs (#55)
- Parallel coordinates and parameter importance for sweeps and AutoML, both apps.
  **Done.** Importance is the correlation ratio per setting (model-free, ignores
  interactions); it needs 4 finished trials.
- Design changes between compared runs. **Done.**

## Phase 3 — bugs and upkeep (#56)
- Fixes: Try it loads no longer block other runs; zip size cap; Elastic Transform's
  sigma checked against the image size; torchvision 0.21 where needed. **Done.**
- Minor dependency updates; the new slow tests listed. **Done.**

## Phase 4 — release 3.6.0 (#57)
- CHANGELOG, README, this roadmap, FAMILIES regenerated, a screenshot of the sweep
  insights, version 3.6.0. The tag and release follow your confirmation.

## Also in 3.6 (#58)
- Every text file is read and written as UTF-8 (about 220 places), so non-ASCII names
  work on Windows with Python < 3.15. `tests/test_encoding.py` keeps it that way.
  **Done.**

## Still open
- TypeScript 7 for the web app (a major version).
- Importance that accounts for interactions between settings (fANOVA), for large sweeps.
