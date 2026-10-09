# Changelog

## Unreleased (3.0 in progress — see docs/ROADMAP_3.0.md)

### Added
- Model families and a task registry: designs are recognised, validated and
  trained through `core/families.py` / `core/tasks.py`; `/api/tasks`,
  `/api/families`, `/api/describe`, MCP `list_tasks`. The status bar shows the
  design's family and task.
- Semantic port roles (logits, boxes, masks, tokens, graph, variable, ...)
  checked on every wire; blocks can declare optional package requirements,
  reported with the pip extra that installs them.
- Trainer-kind registry for non-supervised training loops.
- Pretrained image backbones: 39 torchvision architectures (was 16) from a
  generated, measured catalog — adds ResNet-152, ResNeXt, Wide ResNet,
  EfficientNet-B4 / V2, ConvNeXt S/B, DenseNet-169/201, VGG-19, RegNet-Y
  800MF–3.2GF, ShuffleNetV2, MNASNet, ViT-B/32, ViT-L/16, Swin(V2), MaxViT.

### Fixed
- RegNet backbones were mapped to Keras classes that Keras 3 does not provide;
  they are now reported as PyTorch-only.

## 2.0.0

The designer becomes a complete model workflow: track and tune experiments,
deploy trained models as services, import existing models, inspect data before
training, and work from the browser as well as the desktop.

### Added
- **Experiments.** Every run is recorded in a persistent history
  (`$AIME_HOME/runs`) with its design snapshot, parameters, per-epoch metrics,
  environment (Python, platform, package versions) and data fingerprint.
  The Experiments dock lists, filters, tags, compares (metrics, differing
  parameters, overlaid curves) and restores runs.
- **Hyperparameter sweeps** over any numeric or choice parameter in the design:
  grid, random or Bayesian (Optuna TPE) search, invalid configurations skipped,
  trials recorded as child runs, leaderboard and "Apply best". Deterministic
  training option.
- **Deploy.** Serving packages from a finished run: FastAPI app (`/predict`,
  `/health`, `/metadata`) with the exact fitted preprocessing, the model,
  requirements, Dockerfile and README. Optional ONNX (verified against
  onnxruntime), int8-quantized ONNX, TorchScript and Core ML exports. A model
  registry with versions and staging / production stages.
- **Import.** PyTorch modules (`torch.fx`), ONNX files and Keras models become
  editable designs. Weights are copied into the rebuilt model and outputs are
  compared numerically; unsupported operations are listed.
- **Custom block packs.** Versioned custom blocks, shareable as `.aimeblocks`
  files; `.aime` archives embed the custom blocks they use and restore their
  source run.
- **Data workspace.** Dataset profiles (column statistics, class balance,
  image / text / audio statistics) with findings: missing values, constant and
  identifier columns, target leakage, duplicates, high cardinality, skew,
  broken images. Pipeline-aware data warnings in the Problems panel, an exact
  split preview and an augmentation preview.
- **Web version.** `aime web` serves the designer in the browser (React +
  React Flow) with live validation, training with streamed charts, data,
  experiments and model pages, on a REST + WebSocket API with optional token
  auth. A Docker image is provided.
- Command line: `aime runs`, `sweep`, `deploy`, `models`, `serve`, `import`,
  `data`, `web`. MCP tools for runs, sweeps, deployment, import and data.

### Changed
- One headless API (`core/api.py`) backs the CLI, the MCP server and the web
  server.
- Problems-panel data checks run in the background with a cache instead of
  hashing dataset files on every edit.
- Generated training scripts save the fitted preprocessing state and include
  `load_predictor()` / `infer()` for serving.
- Settings layout version 5 (new Experiments and Data docks); saved window
  layouts from 1.x are reset once.

### Fixed
- Serving inputs that are long base64 strings no longer crash path detection
  (`File name too long`).
- ONNX import verification for models exported with a fixed batch size.
- Several native teardown crashes on CI after successful test sessions.

## 1.0.0

First production release: a professional designer for practitioners.

### Added
- Block library expanded to 345 blocks: complete `torch.nn` layer coverage with
  Keras 3 equivalents, 16 pretrained image backbones, 8 Hugging Face text
  encoders, 55 classic-ML estimators (scikit-learn, XGBoost, LightGBM,
  CatBoost), 12 feature transformers, hyperparameter search.
- Training engine: 12 dataset sources across image, tabular, text, audio and
  time-series data; 45 preprocessing and augmentation blocks; 14 task-aware
  losses, 12 optimizers, 12 schedulers, 25 metrics; mixed precision, gradient
  clipping and accumulation, class balancing, MixUp / CutMix, k-fold CV.
- scikit-learn export target with ColumnTransformer pipelines, CV and search.
- Architecture lints and data/training rules with Quick Fixes.
- Dock-based workbench with dark and light themes, Block Library, Inspector,
  Problems, Output and Training panels, command palette.
- Error analysis, Grad-CAM saliency and model card dialogs.
- Autosave with crash recovery, recent projects, project-file migrations,
  logging and an error reporter, desktop packaging, CI.

### Changed
- Code generation rewritten: independent PyTorch and Keras emitters, layout-aware
  Keras export, readable variable naming, typed documented output.
- Preprocessing statistics are fitted on the training split only.
- Professional diagnostics wording throughout.

### Removed
- Education features: missions, quizzes, prediction gate, celebrations,
  camera / microphone capture, teaching samples.

### Fixed
- Broken Keras export for padded convolutions, bidirectional and stacked RNNs,
  transformer encoders, sequence inputs and image data layout.
- Wrong `in_channels` for Conv1D, binary-classification metrics, BCE target
  shapes, dead branches emitted into `forward()`.
- Blocks with `height` / `width` parameters could not be placed.
- Opening a project marked it as modified; problem jump-to-block crashed.
