# AI Made Easy 3.7 — plan and status

*Status: built in PRs #60–#64, ready for your review. Nothing is merged or tagged until
you say so.*

## Phase 0 — Windows as a tested platform (#60)
- A Windows test job (Python 3.12) and a Windows desktop-bundle job that builds the app
  and checks it starts. **Done.**
- Child processes write UTF-8 to the app (`PYTHONIOENCODING`), and the app decodes UTF-8
  everywhere; a test keeps `text=True` out of the package. **Done.**
- An end-to-end test with an accented folder, columns, classes and design name. **Done.**
- **What the first Windows run found** (243 failures), now fixed: generated scripts
  didn't parse when a Windows data path sat in their docstring (`C:\Users` → `\U`
  escape); `$AIME_DOCKER` / `$AIME_SSH` lost backslashes; export metadata used `\`; a
  run folder could not be deleted while its worker was still exiting; and 270 text reads
  and writes in tests and scripts used the code page. Guards: `tests/test_windows_paths.py`
  and the encoding test, which now covers tests and scripts.
- **Open:** the MCP stdio round-trip test is skipped on Windows: the MCP SDK's stdio
  client crashed the test worker there. The MCP server itself is unchanged.

## Phase 1 — more blocks (#61)
18 layers, 6 losses, 3 optimizers, 3 metrics, 1 augmentation and 2 recipes (see the
CHANGELOG): 31 blocks, 747 in all. **Done**, with these changes to the plan:
- **Dropped:** NGBoost (0.5.11 never marks itself fitted, so scikit-learn pipelines
  refuse to predict with it), t-SNE (no transform for new rows), sMAPE (the forecasting
  family already has it), GridMask (no torchvision transform; a custom one risks
  DataLoader pickling on macOS and Windows).
- **Swapped:** Sophia and SOAP for Cautious AdamW and Yogi (Sophia needs second-order
  passes the training loop doesn't make).
- **Added:** Dynamic Tanh, GRN, ECA, Fused-MBConv, a Hyena operator, an asymmetric
  multi-label loss, concordance, photometric distort.
- Export rules: RWKV and Hyena can't run on ExecuTorch; deformable convolution exports
  only to TorchScript (each checked by trying every format).

## Phase 2 — which settings matter together (#62)
- Importance from functional ANOVA on a random forest of the trials (from 10 trials, with
  scikit-learn), with the strongest pairs and a heatmap of the top pair. **Done.**
  Optuna's fANOVA evaluator reports single settings only, so the decomposition is
  computed from the forest's marginal predictions.
- Sweep again near the best, reviewed in the sweep form, both apps. **Done.**

## Phase 3 — TypeScript 7 and upkeep (#63)
- TypeScript 7.0: no source changes needed; the bundled build is identical. **Done.**
- Every other web dependency was already current.

## Phase 4 — release 3.7.0 (#64)
- CHANGELOG, README, this roadmap, a screenshot of the pair heatmap, version 3.7.0. The
  tag and release follow your confirmation.

## Still open
- The MCP stdio test on Windows (above).
- The slow tests on Windows take most of the CI job's 120 minutes; splitting the job
  would shorten it.
