# Production checklist — AI Made Easy 1.0

Target user: an ML practitioner who knows model design but does not want to
hand-write framework code. Every phase lands on its own branch and is merged
into `main` with a merge commit.

## Phase 0 — Quality baseline ✅
- [x] Fix GLU `NameError`, optimizer detection, duplicated target table
- [x] Python 3.12+ startup (distutils shim), dev deps complete
- [x] Full test suite green (incl. real training runs)

## Phase 1 — Professional product (remove education layer) ✅
- [x] Remove missions / PRIMM stages / quizzes / age tracks
- [x] Remove predict-before-training gate + "surprise" dialog
- [x] Remove celebration confetti, kid copy, emoji-heavy strings
- [x] Remove webcam/microphone capture + bias-arc teaching samples
- [x] Mistake Museum → professional **Error Analysis** (misclassifications, confusion pairs)
- [x] Kid Report Card → standard **Model Card** (metrics, data, intended use)
- [x] Rewrite all user-facing messages in concise technical language
- [x] Tests updated for the removed/renamed features

## Phase 2 — Design-time validation engine ✅
- [x] Wire-time feedback: incompatible ports rejected with a message; new shape errors
      outline the blocks and raise a notification immediately
- [x] Full shape + rank inference across branches/merges with precise messages (expected vs got)
- [x] Dtype propagation (integer index tensors vs float) — Embedding / casting rules
- [x] Architecture lints (`core/lints.py`): double softmax with CrossEntropy, consecutive
      linear layers, activation on regression output, CE with 1 unit, class count vs
      dataset, Dropout before Output, Dense on spatial tensors, huge Dense, stride > kernel,
      attention without positions, BatchNorm with tiny batches, dataset shape vs Input
- [x] Dead branches pruned from the model path (flagged + never exported)
- [x] Professional diagnostic wording (no emoji / teaching tone) with `Fix:` hints
- [x] Problems panel (IDE-style): severity, block, message, click to select
- [x] Every block placeable/editable on the canvas; every model block exercised by runtime tests

## Phase 3 — Block library: every major ML/DL library
- [x] Layers: Linear/Bilinear/Identity, Conv1-3D, ConvTranspose1-3D, Depthwise/Separable,
      Max/Avg/Adaptive/Global/LP pooling, Zero/Reflect/Replicate/Circular padding, Crop,
      Upsample, PixelShuffle, LSTM/GRU/RNN (stacked, bidirectional, final state),
      MHA, cross-attention, Transformer encoder/decoder, Squeeze-Excite, Embedding,
      positional encodings, 27 activations, Batch/Layer/RMS/Group/Instance/LRN norms,
      Dropout family, Gaussian noise, merges, tensor ops
- [x] 14 losses (task-aware: multi-class / binary / multi-label / regression / distribution),
      12 optimizers, 12 LR schedulers (epoch / batch / plateau stepping), 19 metrics
- [x] 12 dataset sources: torchvision (8 benchmarks), image folder, table files
      (CSV/TSV/Parquet/Excel/JSON), scikit-learn datasets, synthetic, NumPy, JSON,
      Hugging Face, text table, text folder, audio folder (WAV), time-series table
- [x] 45 preprocessing blocks: split (stratified / chronological), loader, class balancing,
      impute, one-hot, ordinal, log, outlier clipping, z-score (fit/fixed), min-max, robust,
      variance filter, text cleaning + tokenization (word / char / HF), audio features
      (waveform / spectrogram / mel / MFCC) + SpecAugment, 27 image transforms incl.
      RandAugment / AutoAugment / AugMix / MixUp / CutMix
- [x] All statistics fitted on the training split only (no leakage)
- [x] Trainer: early stopping, gradient clipping, accumulation, mixed precision, k-fold
- [x] Rules: loss ↔ output activation, dataset sample shape ↔ Input, text dtype,
      tokenizer vocab ↔ Embedding, time-series window/horizon ↔ model, modality checks,
      param relations (split fractions, Nesterov, scale ranges, odd kernels, ...)
- [x] End-to-end tests run generated training scripts for every modality (PyTorch + Keras)
- [x] Pretrained models: 16 ImageNet backbones (torchvision + keras.applications) and 8 Hugging Face text encoders, frozen or fine-tuned, with input constraints + normalization rule
- [x] Classic ML family: 55 estimators — linear models, SVMs, neighbors, naive Bayes,
      discriminant analysis, trees & ensembles, sklearn MLPs, clustering (k-Means, DBSCAN,
      HDBSCAN, GMM, ...), anomaly detection; 12 transformers (polynomial, spline, power,
      quantile, binning, SelectKBest, PCA, SVD, kernel PCA, ICA, TF-IDF, counts)
- [x] Gradient boosting: XGBoost, LightGBM, CatBoost (sklearn API)
- [x] scikit-learn target (`sklearn_train`): ColumnTransformer pipelines, CV, grid / random /
      halving search, class balancing, feature importances, joblib export — every estimator
      verified end to end
- [x] Classic rules: one estimator, no mixing with NN layers, non-negative estimators,
      solver/penalty compatibility, score-function vs task, vectorizer vs data, grid syntax
- [x] Keras parity for every NN block that has a Keras equivalent; unsupported blocks
      produce one clear error listing them (never a crash)

## Phase 4 — Code generation quality
- [x] Auto-test: for every NN block (+ risky param variants) generated PyTorch code executes
      and the real output shape equals the designer's inferred shape (177 cases)
- [x] Same auto-test for Keras 3 — layout-tracking emitter (channels-last where needed,
      minimal Permutes), correct padding/momentum/bidirectional/stacked RNN translation
- [x] A Keras gap never blocks PyTorch export; unsupported blocks reported in one error
- [x] Generated code: clean imports, docstrings, type hints, `build_model()`, no dead code
- [x] ONNX / TorchScript exports embed the same verified model source
- [x] sklearn scripts execute on sample data

## Phase 5 — Professional UI ✅
- [x] New design system: neutral dark (default) + light theme, one accent colour, no emoji
- [x] Toolbar with vector icons; grouped actions; command palette kept
- [x] Block library: sectioned tree + search + drag-and-drop, descriptions and library info in tooltips
- [x] Node rendering: header with family accent, port labels, live output shape / key settings,
      error/warning outline
- [x] Dockable panels: Properties · Summary · Code · Assistant (right); Problems · Output · Training (bottom)
- [x] Status bar: validation state, parameter count, device

## Phase 6 — Production readiness ✅
- [x] Version 1.0.0, About dialog, app icon (PNG + macOS .icns)
- [x] Rotating log file (~/.aime/logs) + global exception handler with error dialog
- [x] Autosave every 2 minutes + crash recovery, Open Recent, project-file migrations
- [x] CI: GitHub Actions (ruff + pytest offscreen, Linux + macOS, Python 3.11/3.12)
- [x] Packaging: PyInstaller spec + build script (verified: bundle builds and launches)
- [x] README, CHANGELOG, LICENSE, ARCHITECTURE rewritten for the professional product
- [x] Live notification when a connection introduces a new error

## Phase 7 — Final QA
- [x] Full suite green, lint clean, every sample validates + exports + compiles
- [x] Screenshots of the final UI in README
- [x] Everything merged and pushed to GitHub
- [x] CI green on Ubuntu + macOS, Python 3.11 / 3.12
