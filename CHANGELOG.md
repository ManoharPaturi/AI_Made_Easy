# Changelog

## Unreleased (3.0 in progress — see docs/ROADMAP_3.0.md)

### Added
- Task-first wizard and AutoML (`core/recipes`, `core/automl.py`). New from Task (desktop,
  Ctrl+Shift+N) and New (web): pick one of 29 tasks, point at a table or dataset folder
  (format, target, classes and the fitting tasks are detected: CSV / Parquet / Excel, text
  tables, time series, interactions, edge lists, class folders of images / text / audio,
  COCO, YOLO, VOC, images + masks) or use demo data, set a budget, and get recipes ranked
  for the data size and budget with reasons, cautions and cost estimates. Recipes cover
  every task (MLP, Tabular ResNet, FT-Transformer, random forest, gradient boosting,
  linear baselines, CNNs, ResNet transfer learning, text and audio classifiers, plus
  designs for vision, forecasting, speech, generative, sequence, graph, recommender, RL
  and probabilistic tasks); each is fitted to the data by the safe Quick Fixes and notes
  why every block is there (Why tab, Explain This Design, Properties). AutoML trains every
  candidate once, then tunes recipes and their settings with TPE, prunes weak trials by
  the median rule and skips designs over the budget; results appear with the sweeps.
  `aime new`, `aime automl`, `aime recipes`, `/api/wizard/*`, `/api/automl`, MCP tools.
- Graph neural networks (`core/gnn`, PyTorch Geometric, `graph` extra): GCN, GAT / GATv2,
  GraphSAGE, GIN, GraphConv, EdgeConv and graph transformer layers; global, attention,
  TopK and SAG pooling; a Link Predictor. Node classification, graph classification and
  link prediction (held-out edges, negative sampling, ROC-AUC / AP) on generated community
  or motif graphs, node / edge CSVs, Planetoid and TU benchmarks. Graph models take
  `forward(x, edge_index, batch)`; deploy packages serve node, graph and link requests.
- Tabular deep learning (`core/tabular`): Tabular ResNet, FT-Transformer, TabTransformer,
  TabNet and categorical entity embeddings. A Quick Fix fills the positions and
  cardinalities of ordinal-encoded columns from the data, scaling keeps those codes as
  integers, and a generated Synthetic Table mixes numbers and categories. Modules can
  add an `aux_loss()` to the supervised loss (TabNet's sparsity penalty).
- Recommenders (`core/recsys`): matrix factorisation, NCF, two-tower retrieval and
  DLRM-lite with BPR, BCE (sampled negatives) or MSE objectives; Recall@K, NDCG@K and MRR
  over all items next to a most-popular baseline; top-K serving with a popularity
  fallback; generated interactions (implicit or ratings, with features) and CSV logs.
- Reinforcement learning (`core/rl`, `rl` extra): Gymnasium environments and PPO, A2C,
  DQN, SAC and TD3 from stable-baselines3; the canvas network is the policy's feature
  extractor; live reward curves, best-policy checkpoints, a random-policy baseline, and
  deploy packages that return actions.
- Quick Fixes now also set enum and boolean parameters ("set objective to bpr").
- Samples: GCN node classification, GIN graph classification, GraphSAGE link prediction,
  FT-Transformer churn, Tabular ResNet regression, matrix factorisation, DLRM, PPO on
  CartPole and SAC on Pendulum.
- Probabilistic models, all behind the `probabilistic` extra except the
  PyTorch parts.
  - Graphical models (`core/pgm`, pgmpy): discrete and Gaussian variables wired by a
    multi-input `parents` port; Bayesian and Markov networks, naive Bayes, dynamic
    Bayesian networks and hidden Markov models (hmmlearn). CPD tables are edited in a
    table editor (desktop and web) with a "Normalize the table" Quick Fix. Structure
    learning (GES, Hill-Climb with BIC / BDeu / K2, PC, Chow-Liu trees), MLE /
    Bayesian / EM parameter learning, variable elimination, belief propagation and
    sampling, marginal and MAP queries. Deployed models answer evidence queries.
  - Probabilistic programs (`core/ppl`, PyMC): 19 distribution blocks with a port per
    parameter, deterministic expressions, observed columns and group-level
    (hierarchical) effects, non-centred by default. NUTS (PyMC or NumPyro), ADVI,
    SMC or MAP; split R-hat and bulk / tail ESS identical to ArviZ, divergences,
    trace and posterior-predictive plots. Rules check supports, observed data types
    and improper priors.
  - Gaussian processes (`core/gp`): RBF, Matérn, periodic, linear, rational
    quadratic, spectral-mixture, white and constant kernels combined with Sum /
    Product blocks; exact or sparse variational GPs (GPyTorch) or scikit-learn;
    Gaussian, Student-t, Bernoulli and Poisson likelihoods; interval coverage
    against a constant baseline.
  - Bayesian deep learning (`core/bayes`): MC Dropout, Bayes-by-Backprop linear and
    convolution layers (KL added to the loss), a mixture density head and a
    last-layer Laplace approximation, in any supervised PyTorch design.
  - Calibration: ECE with a reliability diagram, temperature scaling and split
    conformal prediction (class sets or intervals). Served predictions add entropy,
    prediction sets, standard deviations and intervals.
  - Normalizing flows (`core/flows`): RealNVP affine couplings, masked
    autoregressive layers, neural spline couplings, ActNorm and permutations, a
    `density_estimation` task with a maximum-likelihood trainer, 2-D density data
    and CSV tables, a Gaussian baseline and live density pictures. Deployed flows
    return log-densities and samples.
  - State-space models (`core/ssm`, statsmodels): level / trend, seasonal, cycle,
    autoregressive and regression components, or a SARIMAX specification, fitted
    with the Kalman filter; held-out forecasts with intervals against seasonal
    naive, smoothed components, and serving that filters new histories.
  - Samples: student network, structure learning, market-regime HMM, hierarchical
    linear and logistic regression, GP trend + seasonality and GP classification,
    MC-dropout calibration, Bayes by Backprop, a mixture density network, spline and
    RealNVP flows, a structural time series and a seasonal ARIMA.
- Table parameters (`"table"` type, a JSON 2-D list) with grid editors, and
  multi-input ports on the desktop and web canvases.
- Generative models (`core/generative`), each with its own training loop.
  - VAEs built from layers around a Reparameterize bottleneck (β, KL warm-up; MSE
    or binary cross-entropy reconstruction); generation decodes prior samples.
  - GANs: any generator on the canvas (or a DCGAN generator block) against a
    Discriminator block with BCE, hinge or WGAN-GP losses, spectral norm and
    several critic steps.
  - Diffusion: a time- and class-conditioned U-Net with a Noise Scheduler block
    (linear / cosine, DDPM or DDIM sampling, classifier-free guidance, EMA).
  - Language models (GPT block, or Embedding + Causal Transformer) and
    encoder-decoder transformers on byte-level text (text corpora, CSV pairs,
    generated sentences and tasks).
  - Metrics: FID / KID (offline pixel features or Inception-v3), perplexity and
    bits per byte, BLEU / chrF (identical to sacrebleu) and ROUGE-L.
  - Live sample grids and generated text every epoch in the desktop and web
    training panels (a new `samples` worker event); saved samples in the run view.
  - Rules with Quick Fixes: dataset vs Input / generator shape, VAE reconstruction
    shape, latent Input for GANs, U-Net depth vs image size, class counts, causal
    language models (no bidirectional layers), byte vocabularies, context and
    target lengths, one generative model per design.
  - Deploy packages add `POST /generate` (images as base64 PNG, or text).
  - Samples: shapes VAE, DCGAN, class-conditional diffusion, tiny GPT and a
    reverse-the-word seq2seq.
- Time-series forecasting (`core/forecast`), with its own training loop.
  - Models: DLinear, N-BEATS, N-HiTS, PatchTST, TiDE, TCN, a DeepAR-style RNN and
    an Informer-style Transformer, each with a point, quantile (non-crossing),
    Student-t or negative-binomial head.
  - Data: long-format tables with several series (id column), past and
    known-future covariates (promotions, holidays), and generated seasonal demand
    series. Rolling-origin backtest split, per-window scaling.
  - Metrics: MASE (with the seasonal-naive baseline reported alongside), sMAPE,
    WAPE, CRPS, pinball loss, 80% interval coverage and error by horizon step;
    forecast plots with intervals; deploy packages serve forecasts for histories.
  - Rules with Quick Fixes: Input vs window × channels, model vs dataset horizon,
    future-covariate counts, TCN receptive field vs window, head vs data type.
    Profiles report series lengths, gaps, missing values, count data and the
    detected seasonal period.
- Speech (`core/speech`): keyword spotting, speech recognition with CTC and
  audio tagging, with their own training loop.
  - Speech manifests (CSV / TSV of WAV files with keywords, transcripts or tags)
    and generated tone-coded speech; waveform augmentation; CTC loss; accuracy /
    F1, CER / WER and tag mAP; spectrogram samples with predictions. Deploy
    packages accept base64 WAV or waveforms and resample them.
  - Rules: Input vs clip length, class / alphabet counts, CTC frames vs
    transcript length, sample rate vs mel features and 16 kHz pretrained encoders.
- Sequence blocks: TCN, Mamba (selective SSM), S4D and sLSTM (xLSTM) in pure
  PyTorch; in-model mel spectrogram / MFCC front-ends (image or sequence layout);
  wav2vec 2.0, HuBERT, WavLM and Whisper encoders (`audio` extra).
- Samples: demand forecasting (N-HiTS, quantiles), keyword spotting (mel CNN) and
  speech recognition (BiLSTM + CTC).
- Vision tasks (`core/vision`): object detection, instance segmentation, keypoint
  detection and semantic segmentation, with their own training loop and metrics.
  - Models: torchvision Faster / Mask / Keypoint R-CNN, RetinaNet, FCOS, SSD(lite),
    DeepLabV3, FCN, LR-ASPP; Hugging Face DETR, Conditional DETR, RT-DETR and
    SegFormer; a U-Net family (U-Net, U-Net++, Attention U-Net, ResUNet); 87 curated
    timm backbones and DINOv2 / CLIP / SigLIP image encoders.
  - Data: COCO, YOLO (boxes, polygons, pose), Pascal VOC, segmentation mask folders
    and generated Synthetic Shapes. Profiles in the Data workspace (objects per
    class, COCO object sizes, empty images, mask class shares, boxes outside the image).
  - Box / mask / keypoint-aware augmentation (`transforms.v2`); Dice, Tversky,
    Lovász-Softmax and pixel focal losses; COCO mAP@[.5:.95] / mAP@.5 / mask mAP
    (identical to pycocotools), PCK, mean IoU, Dice and pixel accuracy.
  - Design rules with Quick Fixes: class / keypoint counts vs the dataset,
    detectors wired straight from Input to Output, dataset ↔ task compatibility,
    anchors vs image size, SegFormer stride, non-commercial weight licenses.
  - Test-image overlays (truth vs prediction) in Error Analysis (desktop) and
    the run view (web); deploy packages serve boxes, labels, scores and RLE masks.
  - Two samples (Shapes detection, Shapes segmentation); `vision-tasks` extra and CI job.
- Resource budgets (`core/budget.py`): per-layer FLOPs, training memory
  (weights, gradients, optimizer state, saved activations, mixed precision) and
  latency estimates against 11 device profiles (laptop CPU, Apple MPS, RTX
  3060/4090, T4, A100, H100, iPhone ANE, Jetson Orin Nano, Raspberry Pi 5) plus
  your own (`~/.aime/devices.json`). FLOPs match PyTorch's counter within 7%
  and training memory matches measured MPS peaks within 7% on the samples.
- Budget settings (device, max training memory / latency / parameters / model
  size) in the Summary tab (desktop and web) and `meta.budget`; designs over
  budget get warnings with Quick Fixes (mixed precision, or a smaller batch with
  gradient accumulation that keeps the effective batch).
- Sweeps skip trials that break the budget; training runs record measured peak
  memory and step time (`resources` in the run record) to compare with the
  estimate. `aime budget`, `GET /api/devices`, `POST /api/budget`, MCP
  `estimate_budget`; the Summary shows FLOPs per layer.
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
- A training script that stopped with an error message (`SystemExit("...")`, e.g. a data
  shape mismatch) was recorded as a finished run; it is now a failed run with the message.
- Networks reading a table now check their Input width against the preprocessed row
  (one-hot columns included) for every model, not only the tabular blocks.
- The desktop canvas keeps a recipe's per-block notes when it renames blocks on load.
- Desktop training runs now record measured peak memory and step time (they
  were only recorded for runs started from the CLI, web or MCP).
- The "softmax before a logit loss" lint no longer fires on tasks that do not
  train with cross-entropy (VAE, GAN, diffusion, forecasting).
- Deploy packages of vision runs listed no PyTorch requirement: task-specific
  training scripts now declare what they need.
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
