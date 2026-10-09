# AI Made Easy 3.0 — plan

## Context
2.0 is done: experiments, deploy, import, data workspace and the web version, with 345 blocks.
The user wants a much broader V3. They chose:
- **Every model area:** vision tasks, generative, sequences / time series / audio, graph / tabular / RL.
- **Probabilistic models, all of them:** pgmpy graphical models plus probabilistic programming, Gaussian processes and Bayesian deep learning.
- **Workflows:** task-first wizards (with AutoML) and multi-stage pipelines.
- **Constraints:** resource budgets (memory / FLOPs / latency against a target device), plus each new family's own design rules.

What limits growth today:
- Tasks are only classification and regression. `TrainingSpec.task` comes from the loss block's `meta`.
- Project kinds are hardcoded by block prefix in four places:
  - `ui/services/project_service.py::project_kind`
  - `core/runner/manager.py::resolve_framework`
  - `core/classic/generate.py::is_classic`
  - `core/targets.py`
- Ports are fixed and single-input, so graphical models (many parents per variable) don't fit.
- There is one supervised training loop, so GANs, diffusion and RL can't be expressed.

Goal: about 750+ blocks, every family validated at design time, generating runnable code, trainable in-app, deployable, and working in both desktop and web.

**Process (unchanged):**
- One branch per phase, then a PR with CI green.
- Full `pytest` + `ruff` + web build / Playwright before merging.
- Heavy optional libraries go behind extras.
- Tests use tiny synthetic data and never download weights.

---

## Phase 0 — Families, tasks and port roles (the foundation)
- **`core/families.py`:** a `Family` registry.
  - Each family declares:
    - id and label
    - block prefixes and detection
    - extra validators
    - codegen targets (registered via the existing `core/targets.register_target`)
    - training runner
    - summary provider
    - serving adapter
    - required extras
  - Port `neural`, `classic` and `llm` to it.
  - Replace the four hardcoded checks with `family_of(graph)`.
- **`core/tasks.py`:** a `Task` registry. Each task declares:
  - target format and output port role
  - compatible heads, losses and metrics
  - dataset formats and augmentation kind (image / box-aware / mask-aware / sequence)
  - serving request/response schema and evaluation views

  Existing tasks move into it: multiclass, binary, multilabel, regression, distribution. `TrainingSpec.task` then resolves through the task registry instead of loss `meta` alone.
- **Port roles and multiplicity in `core/spec.py` (`PortSpec`):**
  - `role` ∈ tensor, logits, probs, boxes, masks, keypoints, tokens, graph, distribution, variable, latent.
  - `multi=True` lets one input port accept many edges.
  - Graph IR and validation enforce both. Canvas and web render multi-ports.
- **Trainer kinds:** `supervised` (current), `adversarial`, `diffusion`, `vae`, `self_supervised`, `rl`, `bayesian`.
  - The training template is chosen by trainer kind.
  - The worker event protocol stays the same (`epoch` / `log` / `done`).
- **Optional-dependency gating:**
  - Blocks declare `requires=("timm",)`.
  - The library shows an "install X" badge; validation reports a clear error with the pip extra.
- **Model-zoo generator** (`scripts/build_zoo.py`): curated pretrained catalogs (timm, HF vision / audio / text) become JSON option lists with license, parameter count and input size. No network at runtime.
- **Per-family test harness:** generalize `tests/test_codegen_runtime.py` so every family runs "every block generates, executes and gives the designed shape".

## Phase 1 — Resource budgets (constraints)
- **`core/budget.py`:**
  - Per-block FLOPs and activation-memory estimates. Formulas by layer kind (conv / linear / attention / norm / pool), extending `summary.py` (which already has param counts and `param_fn`).
  - Training memory = params + grads + optimizer states (Adam ×2) + activations × batch. Accounts for AMP, gradient checkpointing and accumulation.
  - Inference latency estimate = FLOPs / device throughput + a memory-bandwidth term.
- **Device profiles** (`core/devices.json`): CPU laptop, Apple M-series (MPS), RTX 3060 / 4090, T4, A100 40/80, H100, iPhone ANE, Jetson Orin Nano, Raspberry Pi 5. Users can add their own.
- **Budget setting** (project-level, shown in the Trainer inspector / status bar): device, max train memory, max inference latency, max params / model size.
- **Lints with Quick Fixes:**
  - "won't fit in 8 GB at batch 64 (est. 11.2 GB)" → reduce batch / enable AMP / gradient checkpointing / accumulation
  - latency over budget
  - model size over a deploy limit
- **Calibration:** the first epoch of a real run records measured peak memory and step time (`torch.cuda.max_memory_allocated` / MPS / wall time). These go into the run record, and the Summary shows estimate vs measured to tune the coefficients.
- **UI:**
  - Summary tab gains FLOPs / memory / latency columns and a budget bar.
  - The web Summary gets the same.
  - The sweep dialog can skip trials that exceed the budget.

## Phase 2 — Vision tasks
- **Tasks:**
  - object detection
  - semantic segmentation
  - instance segmentation
  - keypoints
  - image regression / multilabel (already partly there)
- **Models:**
  - torchvision detectors: Faster / Mask / Keypoint R-CNN, RetinaNet, FCOS, SSD(lite)
  - segmentation: DeepLabV3(+), FCN, LR-ASPP, U-Net family (U-Net++, Attention U-Net, ResUNet)
  - HF transformers: DETR / RT-DETR, SegFormer, Mask2Former
  - backbones: timm (curated ~80: ConvNeXt, EfficientNetV2, RegNet, Swin, ViT, DeiT, MaxViT, MobileNetV4, EVA) and HF (DINOv2, CLIP / SigLIP image towers)
  - FPN neck, detection / segmentation heads as blocks
  - Ultralytics YOLO is excluded (AGPL); a permissive YOLO-style detector is composed from blocks instead.
- **Data:** COCO JSON, YOLO txt, Pascal VOC XML, mask folders. Profiles show box / mask stats (object sizes, class balance per object, empty images).
- **Augmentation:** box- and mask-aware via `torchvision.transforms.v2` tv_tensors. The augmentation preview draws boxes and masks.
- **Losses / metrics:**
  - losses: focal, GIoU / DIoU, Dice, Tversky, Lovász
  - metrics: mAP@[.5:.95] (torchmetrics), IoU / mIoU, Dice, PCK
- **Rules:**
  - number of classes incl. background matches the dataset
  - input size divisible by the backbone stride
  - anchor sizes vs image size
  - box-format consistency
  - mask channels = classes
- **Results / serving:**
  - error analysis shows predicted vs true boxes and masks
  - `/predict` returns boxes / labels / scores and RLE masks; the deploy templates are extended
  - ONNX export notes for detectors
- **Samples:** a detection sample (tiny synthetic shapes dataset) and a segmentation sample.

## Phase 3 — Sequences, time series and audio
- **Forecasting:**
  - TCN, N-BEATS, N-HiTS, PatchTST, DLinear, TiDE, Informer-style attention, TFT-lite
  - probabilistic heads: quantile, Student-t / NegBin DeepAR-style
  - multi-horizon windows, covariates, multiple series (id column), backtesting split
  - metrics: MASE, sMAPE, WAPE, CRPS, pinball
- **Sequence models:**
  - Mamba / S4-style SSM blocks (pure PyTorch)
  - GRU / LSTM variants (bidirectional, peephole-free, projection)
  - xLSTM-lite
  - CTC loss
- **Audio:**
  - HF wav2vec2 / HuBERT / Whisper encoder / AST backbones
  - tasks: speech classification, keyword spotting, ASR (CTC), audio tagging
  - torchaudio features: MelSpectrogram / MFCC on GPU
  - metrics: WER / CER
- **Rules:**
  - window > horizon
  - receptive field ≥ window for TCN
  - sampling-rate match with the pretrained audio model
  - CTC: output length ≥ label length
- **Samples:** forecasting (synthetic seasonal series), keyword spotting (synthetic tones).

## Phase 4 — Generative models
- **Families:**
  - VAE / β-VAE: reparameterize block, KL loss
  - GAN, DCGAN-style: generator + discriminator subgraphs; the adversarial trainer has two optimizers and supports WGAN-GP
  - diffusion: UNet2D with time embedding, DDPM / DDIM schedulers, classifier-free guidance
  - autoregressive / causal LM trained from scratch
  - seq2seq encoder-decoder (translation / summarization)
  - LoRA fine-tuning of Stable Diffusion via `diffusers` (optional)
- **Canvas:** multi-model designs use named sub-graphs ("generator", "discriminator", "denoiser"), represented as composite blocks.
- **Rules:**
  - generator output shape = discriminator input shape
  - latent dim consistency
  - diffusion image size divisible by 2^depth
  - timestep embedding is wired
  - causal mask present for autoregressive models
- **Metrics / results:**
  - metrics: FID / IS / KID (torchmetrics), BLEU / ROUGE / chrF, perplexity
  - a sample grid after each epoch (worker `samples` event), shown in the Training panel and the web
- **Serving:** `/generate` endpoints (seed, steps, guidance, prompt).

## Phase 5 — Probabilistic models (new families)
- **Graphical models (pgmpy):**
  - New family `pgm.*` on the same canvas: nodes are random variables, edges are dependencies (multi-input `parents` port).
  - Variable blocks: discrete with state names, continuous Gaussian.
  - Bayesian Network, Markov Network, Dynamic BN, Naive Bayes, HMM (`hmmlearn`).
  - CPD / factor tables edited in a table editor in the inspector (desktop and web).
  - Structure learning: Hill-Climb with BIC / K2 / BDeu, PC, MMHC, tree / Chow-Liu.
  - Parameter learning: MLE, Bayesian (Dirichlet priors), EM for latent variables.
  - Inference: Variable Elimination, Belief Propagation, sampling.
  - Query / evidence / MAP blocks.
  - Codegen: a pgmpy script; results show posteriors / CPDs.
  - Rules:
    - the BN is a DAG
    - CPD shape = card × ∏ parent cards
    - columns sum to 1
    - evidence states exist
    - unconnected variables
    - cardinality explosion warning (counts as budget)
- **Probabilistic programming (PyMC, NumPyro optional):**
  - Distribution blocks: Normal, HalfNormal, StudentT, Beta, Gamma, Exponential, Poisson, NegBinomial, Bernoulli, Binomial, Categorical, Dirichlet, MvNormal, LKJ, ...
  - Deterministic expression / link-function blocks; observed-data binding to a dataset column.
  - Hierarchical (group index) plates.
  - Inference: NUTS / ADVI / SMC.
  - ArviZ diagnostics: R-hat, ESS, divergences, trace, posterior predictive.
  - Rules:
    - support constraints (scale > 0 needs a positive distribution)
    - shape broadcasting across plates
    - observed dtype matches the likelihood
    - improper priors
- **Gaussian processes:**
  - GPyTorch exact / approximate (SVGP) and the sklearn GP
  - kernel blocks combined by sum / product merges: RBF, Matérn, Periodic, Linear, RationalQuadratic, White, Spectral Mixture
  - likelihoods: Gaussian, Bernoulli, Poisson
- **Bayesian deep learning:**
  - MC Dropout
  - Bayes-by-Backprop Linear / Conv (variational layers)
  - Laplace approximation
  - Mixture Density Network head
  - normalizing flows: RealNVP / MAF / NSF coupling blocks (shared with Phase 4)
  - deep ensembles via Phase 8 pipelines
- **Calibration and uncertainty:**
  - evaluation blocks: temperature scaling, reliability diagram / ECE, conformal prediction (split / CQR)
  - prediction intervals carried into serving responses
- **Kalman / state-space models:** statsmodels UnobservedComponents, Kalman filter blocks for time series.

## Phase 6 — Graph, tabular deep learning, recommenders, RL
- **Graph (PyTorch Geometric, PyTorch only):**
  - GCN, GAT(v2), GraphSAGE, GIN, GraphConv, EdgeConv, TransformerConv
  - global mean / max / add / attention pooling, TopK / SAG pooling
  - tasks: node classification, graph classification, link prediction
  - data: Planetoid, TUDataset, OGB (optional), edge-list + node-feature CSV
  - rules: node-feature dim, edge-index presence, isolated-node warnings
- **Tabular deep learning:**
  - TabNet, FT-Transformer, TabTransformer, ResNet-MLP
  - categorical entity embeddings fed by the data profile (cardinalities)
- **Recommenders:** matrix factorization, NCF, two-tower retrieval, DLRM-lite; metrics: NDCG@k, Recall@k, MRR.
- **Reinforcement learning (Gymnasium + stable-baselines3):**
  - environment block, algorithm blocks (DQN, PPO, A2C, SAC, TD3)
  - the policy / value network is designed on the canvas and becomes an SB3 custom policy
  - reward curve events; evaluation episodes
  - rules: action-space vs head output (discrete vs continuous)

## Phase 7 — Task-first wizards and AutoML
- **New-project wizard (desktop and web):**
  1. Pick a task (from the Phase 0 task registry).
  2. Point at data. The data profile auto-detects the format (COCO / YOLO / mask folders / table / series / graph) and the target.
  3. Set a budget (Phase 1 device profile).
  4. Get a ranked list of **recipes**.
- **Recipes** (`core/recipes/`): parameterized design templates per task × modality × budget tier (small / medium / large / pretrained). Each recipe has a baseline design, augmentation, loss, metrics and trainer defaults, and is validated by the existing graph validation.
- **AutoML:**
  - a sweep over recipe choice (architecture as a choice dimension) and key hyperparameters, using `core/sweeps.py` (Optuna TPE)
  - early-stopping trials (Optuna pruner fed by epoch events)
  - budget-filtered; the leaderboard is reused from the Experiments page
- "Explain this design" panel: why each block was chosen, from recipe metadata.

## Phase 8 — Multi-stage pipelines
- **Family `pipeline.*`.** Stage blocks on the canvas connected in order. Each references a design (embedded or a project file).
  - Stages: Train, Fine-tune (freeze / unfreeze schedule, discriminative learning rates), Distill (teacher → student, KD loss), Ensemble (average / vote / stacking), Cross-validate (k-fold), Prune, Quantization-aware training, Evaluate (test set + calibration), Export, Deploy / Register.
  - Artifacts (checkpoints, metrics) flow along the edges.
- **`core/pipelines.py`: PipelineRunner** on top of `RunManager`:
  - each stage is a child run (`parent` = pipeline run)
  - stages are resumable from the last finished stage
  - stages are cached by design + data fingerprint (an unchanged stage is skipped)
- **Rules:**
  - teacher / student output compatibility
  - the fine-tune stage references a checkpoint whose architecture matches
  - ensemble members share the output role
  - the pipeline budget is the sum of its stages
- **UI:**
  - pipeline progress view (stage timeline)
  - Experiments groups pipeline runs
  - web page for pipelines
  - `aime pipeline run|status`, plus MCP tools

## Phase 9 — Release 3.0.0
- Project schema v3 migration (port roles, multi-ports, family / task fields) in `core/migrate.py`, with old projects loading unchanged.
- **CI:**
  - a test job per extra group (vision, generative, audio, graph, probabilistic, rl) to keep run time bounded
  - nightly job with downloads / pretrained weights
- Web parity audit: CPD editor, sample grids, box / mask viewers, pipeline view, budget bar.
- Docs:
  - README, ARCHITECTURE (families / tasks)
  - new `docs/FAMILIES.md` with every block family and its rules
  - CHANGELOG, screenshots
- Version 3.0.0; check the PyInstaller bundle and Docker image (CPU) with the new extras.

---

## Extras (pyproject)
| Extra | Packages |
| --- | --- |
| `vision-tasks` | timm, torchmetrics[detection], pycocotools |
| `generative` | diffusers, torchmetrics[image] |
| `audio` | torchaudio, transformers, jiwer |
| `graph` | torch_geometric |
| `probabilistic` | pgmpy, pymc, arviz, numpyro (optional), gpytorch, hmmlearn, statsmodels |
| `rl` | gymnasium, stable-baselines3 |

## Reuse
- Validation / lints: `core/graph.py` validate and `core/lints.py` RULES, with Quick Fixes in `core/fixes.py`.
- Sweeps and leaderboard: `core/sweeps.py`, the Experiments UI.
- Data profiling / format detection: `core/data/profile.py`, `lints.py`, `augment.py`.
- Runs: `core/runner/manager.py` and `core/runs/history.py`.
- Deploy: `core/deploy/` templates (add task-specific request / response schemas).
- Codegen targets: `core/targets.register_target`.
- Training templates: `core/training/*_template.py` (add trainer kinds alongside).
- Web: block catalog, inspector and validation are already registry-driven. New editors are added as inspector field types.

## Verification (every phase)
- `ruff` + full `pytest` + `npm run build` + Playwright, locally and on CI (all jobs green) before merging the PR.
- **Per new block:** generated code executes and gives the designed output shape (per-family runtime harness).
- **Per new task / family:** a tiny synthetic end-to-end training run, an inference check through the serving package, a sample project, and a web smoke step.
- **Phase 1:** budget estimates are within ±30% of measured peak memory / step time for reference models (MLP, ResNet-18, small transformer) on CPU / MPS.
- **Phase 5:** pgmpy queries on the textbook "student" network match known posteriors; a PyMC linear regression recovers the true coefficients; GP regression beats a constant baseline.
- **Phase 7:** the wizard produces a valid, trainable design for every task; AutoML on synthetic data improves on the baseline within its budget.
- **Phase 8:** a train → distill → quantize → deploy pipeline runs end to end and resumes after interruption.
