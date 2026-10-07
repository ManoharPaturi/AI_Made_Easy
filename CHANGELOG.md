# Changelog

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
