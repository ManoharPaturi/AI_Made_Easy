# Production checklist — AI Made Easy 1.0

Target user: an ML practitioner who knows model design but does not want to
hand-write framework code. Every phase lands on its own branch and is merged
into `main` with a merge commit.

## Phase 0 — Quality baseline ✅
- [x] Fix GLU `NameError`, optimizer detection, duplicated target table
- [x] Python 3.12+ startup (distutils shim), dev deps complete
- [x] Full test suite green (incl. real training runs)

## Phase 1 — Professional product (remove education layer)
- [ ] Remove missions / PRIMM stages / quizzes / age tracks
- [ ] Remove predict-before-training gate + "surprise" dialog
- [ ] Remove celebration confetti, kid copy, emoji-heavy strings
- [ ] Remove webcam/microphone capture + bias-arc teaching samples
- [ ] Mistake Museum → professional **Error Analysis** (misclassifications, confusion pairs)
- [ ] Kid Report Card → standard **Model Card** (metrics, data, intended use)
- [ ] Rewrite all user-facing messages in concise technical language
- [ ] Tests updated for the removed/renamed features

## Phase 2 — Design-time validation engine
- [ ] Wire-time compatibility check: incompatible connection → inline warning on both blocks
- [x] Full shape + rank inference across branches/merges with precise messages (expected vs got)
- [x] Dtype propagation (integer index tensors vs float) — Embedding / casting rules
- [x] Architecture lints (`core/lints.py`): double softmax with CrossEntropy, consecutive
      linear layers, activation on regression output, CE with 1 unit, class count vs
      dataset, Dropout before Output, Dense on spatial tensors, huge Dense, stride > kernel,
      attention without positions, BatchNorm with tiny batches, dataset shape vs Input
- [x] Dead branches pruned from the model path (flagged + never exported)
- [x] Professional diagnostic wording (no emoji / teaching tone) with `Fix:` hints
- [ ] Problems panel (IDE-style): severity, block, message, click to select
- [ ] Every block has shape inference; registry audit test enforces it

## Phase 3 — Block library: every major ML/DL library
- [x] Layers: Linear/Bilinear/Identity, Conv1-3D, ConvTranspose1-3D, Depthwise/Separable,
      Max/Avg/Adaptive/Global/LP pooling, Zero/Reflect/Replicate/Circular padding, Crop,
      Upsample, PixelShuffle, LSTM/GRU/RNN (stacked, bidirectional, final state),
      MHA, cross-attention, Transformer encoder/decoder, Squeeze-Excite, Embedding,
      positional encodings, 27 activations, Batch/Layer/RMS/Group/Instance/LRN norms,
      Dropout family, Gaussian noise, merges, tensor ops
- [ ] Losses (all torch.nn losses), optimizers, schedulers
- [ ] Datasets per modality (image / text / tabular / audio / time series) with every
      common preprocessing + augmentation method, and per-block rules & constraints
- [ ] Pretrained backbones (torchvision): ResNet, MobileNet, EfficientNet, ConvNeXt, ViT, …
- [ ] Classic ML family (scikit-learn): preprocessing, linear models, SVM, trees, ensembles,
      neighbors, naive Bayes, clustering, decomposition, model selection
- [ ] Gradient boosting: XGBoost, LightGBM, CatBoost (sklearn API)
- [ ] scikit-learn code generation target (`sklearn_train`), runnable end-to-end
- [ ] Keras parity for every NN block that has a Keras equivalent; unsupported blocks
      produce one clear error listing them (never a crash)

## Phase 4 — Code generation quality
- [x] Auto-test: for every NN block (+ risky param variants) generated PyTorch code executes
      and the real output shape equals the designer's inferred shape (177 cases)
- [x] Same auto-test for Keras 3 — layout-tracking emitter (channels-last where needed,
      minimal Permutes), correct padding/momentum/bidirectional/stacked RNN translation
- [x] A Keras gap never blocks PyTorch export; unsupported blocks reported in one error
- [x] Generated code: clean imports, docstrings, type hints, `build_model()`, no dead code
- [x] ONNX / TorchScript exports embed the same verified model source
- [ ] sklearn scripts execute on sample data

## Phase 5 — Professional UI
- [ ] New design system: neutral dark (default) + light theme, one accent colour, no emoji
- [ ] Toolbar with vector icons; grouped actions; command palette kept
- [ ] Block library: categorised tree + search, library badges (PyTorch / Keras / sklearn)
- [ ] Node rendering: header with category accent, port labels, live output-shape badge,
      error/warning outline
- [ ] Panels: Properties · Model Summary · Code (right); Problems · Output · Training (bottom)
- [ ] Status bar: validation state, parameter count, device

## Phase 6 — Production readiness
- [ ] Version 1.0.0, About dialog, app icon
- [ ] Rotating log file + global exception handler (error dialog, no silent crashes)
- [ ] Autosave + crash recovery, recent files, project format versioning
- [ ] CI: GitHub Actions (ruff + pytest offscreen) on every push/PR
- [ ] Packaging: PyInstaller spec + build script for a desktop bundle
- [ ] README, CHANGELOG, architecture docs rewritten for the professional product

## Phase 7 — Final QA
- [ ] Full suite green, lint clean, every sample validates + exports + compiles
- [ ] Screenshots of the final UI in README
- [ ] Everything merged and pushed to GitHub
