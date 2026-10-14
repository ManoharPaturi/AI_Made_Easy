# AI Made Easy 3.2 — plan

## Context
3.1 shipped 9 families, 35 tasks, 59 recipes and 544 blocks.
- Every design has a task.
- Every family's recipes train on the user's own data.
- AutoML can trade accuracy against latency or size.

3.0 and 3.1 widened what you can design. 3.2 makes it solid and gets designs out of
the app. It starts with a bug and CI pass. Then it sends trained models onto devices and
into the tools teams already use (MLflow, W&B, Hugging Face, Kaggle). It also trains on
machines other than the laptop: SSH hosts, Kaggle, Colab and Modal. It closes the last
items 3.0 put off.

Gaps that most limit users today, in order:
1. **Reliability.**
   - CI jobs can hang. The 3.1 release had a macOS 3.11 job stuck with no timeout.
   - Nothing checks every sample and recipe end to end on every OS.
   - Large designs (500+ nodes) are untested for speed.
2. **Training happens only on the machine running the app.** Users with a GPU box, a
   Kaggle account or Colab can't train there from AI Made Easy.
3. **Some trained models can't leave the app as files.**
   - Detectors and graph networks deploy only as the FastAPI package.
   - There is no TFLite / ExecuTorch export for the phones and boards the budget
     profiles describe.
4. **Runs and models stay inside AI Made Easy.** There is no MLflow, Weights & Biases,
   Hugging Face Hub or Kaggle link.
5. **AutoML stops at neural networks and supervised metrics.**
   - Multi-objective search skips classic recipes.
   - Probabilistic tasks have no AutoML metric.
6. **A few deferred model items remain:**
   - Glow
   - OGB datasets
   - Atari
   - design-time checks of custom environments
   - quantization-aware training (QAT) for token-id models

**Credentials rule (applies to every integration):** AI Made Easy never asks for,
stores or logs a password, token or API key. It uses each tool's own sign-in:
- `~/.ssh/config` and keys
- `~/.kaggle/kaggle.json`
- `huggingface-cli login`
- `modal token`
- `wandb login`
- Google Drive for desktop (for Colab runs)

Settings only say *which* host, account or project to use. Every integration is off
until the user turns it on.

**Process (unchanged from 3.1):**
- One branch and one PR per phase, with CI green.
- Full `pytest` + `ruff` + web build / Playwright before merging.
- Heavy libraries go behind extras.
- Tests use tiny synthetic data. Real accounts, weights and downloads are used only in
  the nightly `AIME_ONLINE` job (or are mocked).

---

## Phase 0 — Bugs and CI reliability
- **CI:**
  - a `timeout-minutes` on every job, and a per-step timeout on the Playwright and
    nightly steps
  - pytest-timeout so one hung test fails fast with a traceback instead of hanging
  - find out why the macOS 3.11 job hung (logs from the 3.1 release run), fix the cause,
    and keep a retry only for network installs
  - a test-time budget per job, with slow tests moved to the extras jobs
- **End-to-end sweep (new `scripts/sweep_e2e.py`, run in CI on Linux and in the nightly
  job on every OS):** every sample and every recipe loads, migrates, validates with no
  errors, and generates code. It also trains one tiny epoch and serves one prediction
  from its deploy package. Every failure is filed and fixed in this phase.
- **Bug hunt:**
  - desktop and web smoke runs of each menu / page against the current `main`
  - fix what breaks, each with a regression test
  - Known-bug list kept in this doc's status line.
- **Large designs:**
  - a 500-node benchmark design
  - validation, shape inference and canvas redraw stay under a time budget on desktop
    and web, as a test, with the slow paths profiled and fixed
- **Status:** done.
  - **CI hangs:**
    - Every job has a timeout. pytest-timeout (900 s, thread method) fails a hung test
      with every thread's stack.
    - The 3.1 macOS hang stopped past 93% with orphaned worker processes and did not
      reproduce locally.
  - **Flaky tests:**
    - The QAT tolerance is now 2 points on a 358-digit test split.
    - The server test prints the worker's log when a run ends early.
    - The nightly text-encoder failure was already fixed in `bbd6fea`.
  - **`scripts/sweep_e2e.py`** loads, validates and generates code for all 56 samples
    and 68 recipe / task pairs. With `--train --deploy` it trains each one and builds a
    verified package. CI runs the quick mode; nightly runs everything online.
  - **Bugs it found:**
    - DLRM packages failed verification (integer dummy input for a float row).
    - Class-conditional diffusion packages failed: the export check called the U-Net
      with one tensor even when no export format was asked for.
    - VAE / GAN / diffusion packages were described as image classifiers, so they had
      no `/generate` route.
    - Torchvision downloads had no socket timeout: a dead network hung the run forever.
    - Stable Diffusion LoRA with weights = none crashed at the default 512 px: the tiny
      offline model's attention asked for hundreds of GiB. It now works at up to 64 px.
    - Stable Diffusion LoRA runs could not be deployed (no `*_best.pt`; the packager now
      takes the adapters) and were described as image classifiers (now `/generate`).
  - Sweep runs shrink image and audio sets to 64 items, so Mask R-CNN and a randomly
    initialised wav2vec2 finish in minutes instead of timing out.
  - **Large designs:**
    - Validating 1000 blocks took 292 ms and now takes 20 ms (an edge index per pass).
    - Opening 500 blocks on the desktop took 22 s and now takes about 0.3 s, through
      faster view centering, an x-sorted overlap check, O(1) node lookup, no undo
      entries while loading, linear unique names, and device detection off the UI
      thread.
    - Auto layout is now one undo step, and none when a project opens.
    - The web opens 500 blocks in about 1 s.

## Phase 1 — Train anywhere
- **One runner interface** (`core/runner/backends/`) with the same worker events
  (`epoch` / `log` / `done`). Live curves, stop, resume, sweeps and pipelines then work
  on every backend. Backends:
  - **Local** (today's runner).
  - **SSH:**
    - any host in `~/.ssh/config`, through the system `ssh` client (keys, agent and
      jump hosts as in a terminal; it never prompts for a password)
    - the run travels as a bundle over the same connection (tar, no rsync needed)
    - the training runs under `nohup`, so a dropped connection loses nothing: the app
      reconnects and continues from the last event it received
    - The same backend covers GPU clouds that hand out SSH machines (RunPod, Lambda,
      Vast.ai, Paperspace), with a short guide for each.
  - **Kaggle** (Kaggle API, `kaggle` package):
    - pushes the run as a private Kaggle notebook (script kernel) with T4 / P100 / TPU
    - attaches Kaggle datasets
    - polls the kernel status
    - downloads the outputs into the run folder when it finishes
    - Kaggle has no live log API, so curves arrive when the kernel finishes, and the UI
      says so.
  - **Colab:**
    - Colab has no job API, so this is a hand-off: AI Made Easy builds a notebook with
      the run bundle
    - with Google Drive for desktop, the bundle and notebook go to `My Drive/AI Made
      Easy`; the notebook writes events and checkpoints back there, and the app follows
      them as Drive syncs
    - without it, the notebook ends by downloading a results zip, which **Import
      results** / `aime runs import` turns into the run record
    - (Changed from the first plan: the Drive API with the `drive.file` scope cannot see
      files the Colab notebook writes, and a wider scope would mean holding a broad
      Google token. Drive for desktop needs no token in AI Made Easy at all.)
  - **Modal:**
    - runs the launcher in a Modal sandbox on a Modal GPU, with an image built from the
      run's requirements
    - logs stream live
    - the bundle and the outputs pass through a Modal volume
  - **Container:** an image built like the deploy Dockerfile's (or one you name), on
    this computer, or on an SSH host through its `docker_image` option.
- **Budgets:** each backend reports its accelerator (SSH: `nvidia-smi`; Kaggle: T4 /
  P100 / TPU; Colab / Modal: the chosen GPU). That picks the device profile (P100, L4
  and A10G are new), and the design's budget is checked against it when the run
  starts.
- **UI:**
  - a **Run on** selector in the Training panel and the sweep / pipeline dialogs
    (desktop and web)
  - a Backends dialog to add backends and test the connection (desktop **Run ▸
    Backends…**, web **Run on ▸ Manage backends…**)
  - `aime run --on ssh:<host>|kaggle|colab|modal|container`, `aime backends`,
    `aime runs attach | import`
  - MCP tools and REST routes
- **Tests:**
  - an SSH server in a CI container (localhost) runs a tiny training end to end,
    including a dropped connection
  - Kaggle and Modal clients are faked at their SDK boundary in CI; Colab syncs through
    a folder that stands in for Drive
  - the nightly `AIME_ONLINE` job runs one real Kaggle kernel when a token is
    configured as a secret
- **Status: done.**
  - Everything runs through one launcher, `worker/aime_launch.py`, which is standard
    library only. It unpacks the bundle, trains, writes `events.jsonl`, and always
    leaves an exit code.
  - **The bundle** carries the script, the worker, the requirements and every local
    file the design points at. Paths are rewritten to `data/…`, and nothing else from
    the computer is sent.
  - **SSH:**
    - its runs match a local run with the same seed, metric for metric
    - a cut connection resumes with no epoch lost or repeated
    - after a restart, `aime runs attach` (or **Follow again**) picks the run up again
  - The SSH and Docker tests run the shipped code against stand-in `ssh` and `docker`
    clients. CI also runs the SSH test against real OpenSSH on localhost.
  - **Bugs found while testing:**
    - starting an SSH run blocked until training ended (`cd && nohup … &` backgrounds
      a subshell that keeps ssh's output open)
    - files written into the bundle had a 1970 timestamp, which Colab's results zip
      refuses
    - the web Training panel missed a run's first log lines, including the hand-off
      instructions; it now replays them
  - **Not done:** a Modal run cannot be followed again after the app restarts. Its
    outputs stay in the `aime-runs` volume.

## Phase 2 — Export everywhere
- **Detectors to ONNX:**
  - RetinaNet / FCOS / SSD / Faster R-CNN through torchvision's exportable path (fixed
    input size, post-processing included)
  - DETR / RT-DETR through Hugging Face's exporter
  - Models that can't be exported get a design rule that names the reason.
- **Graph networks to ONNX:** GCN, GraphSAGE, GIN and GAT with static `edge_index`
  shapes, via PyG's ONNX support.
- **On-device:**
  - ExecuTorch (`.pte`) for PyTorch designs
  - LiteRT / TFLite for Keras designs
  - both with int8 post-training quantization from a calibration split
  - Device profiles suggest the matching format.
- **Pipelines:** the Export stage gains these formats. Its metrics compare the exported
  model's outputs with PyTorch.
- **Tests:** every exportable sample round-trips through onnxruntime / ExecuTorch /
  LiteRT and matches PyTorch within a tolerance.

## Phase 3 — Work with your tools
- **Experiment tracking:**
  - **MLflow** and **Weights & Biases** behind one logger interface
  - parameters, per-epoch metrics, the design JSON and artifacts are mirrored as each
    run goes, on every Phase 1 backend
  - set per project or with `AIME_TRACKING`
- **Model registries:**
  - promoting a version can also register it in MLflow's model registry or as a W&B
    artifact
  - an MLflow PyTorch / ONNX model opens as a design where the importers allow
- **Hugging Face Hub:**
  - push a trained run as a model repo, with the model card AI Made Easy already
    generates
  - deploy the FastAPI package as a Docker Space
  - load Hub datasets as a dataset block
- **Kaggle datasets:** a dataset block that downloads a Kaggle dataset or competition
  file and goes through data detection like a local folder. Optionally, submit
  predictions to a competition (confirmed in the UI each time).
- **Run records:** `aime runs export` writes CSV / Parquet for notebooks.
- **Tests:**
  - MLflow with a local file store
  - W&B in offline mode
  - Hub and Kaggle clients mocked; real calls only in the nightly job

## Phase 4 — AutoML for every family
- **Multi-objective for classic ML:**
  - cost estimates for trees, boosting and linear models: model size, plus predict
    latency measured on the host and scaled to the device
  - one Pareto front then mixes classic and neural recipes
- **Metrics for probabilistic tasks:**

  | Family | Metric |
  | --- | --- |
  | PGM / HMM | held-out log-likelihood |
  | PyMC | PSIS-LOO ELPD |
  | GP | test negative log predictive density |
  | State-space | backtest MASE |
  | Flows | bits per dimension |

- **Budgets for classic and probabilistic designs:** memory estimates (trees × depth,
  CPD table sizes, posterior draws).
- **Remote trials:** AutoML trials can run on a Phase 1 backend, with a concurrency
  limit.

## Phase 5 — The last deferred items
- **Glow:** actnorm, invertible 1×1 conv, affine coupling, squeeze / split; image bits
  per dimension.
- **OGB:** node / graph / link property datasets with their official evaluators. They
  download only in the nightly job.
- **Atari:**
  - opt-in environments through `ale-py` (new `atari` extra), with frame stacking /
    preprocessing blocks and the 3.1 CNN policies
  - ROMs are never bundled. The extra's install note tells the user how to add them,
    and a rule says when they are missing.
- **Custom environments:** observation / action spaces are read at design time, so the
  checks move into the Problems panel.
- **QAT for embedding models:** embeddings keep float qparams, and the rest quantizes.
  This lifts the "token-id models are not supported" rule.

## Phase 6 — Release 3.2.0
- **Web parity audit:**
  - Run on / Backends page
  - export formats
  - tracking and Hub settings
- **Docs:**
  - README, ARCHITECTURE (runner backends, export layer)
  - new `docs/DEPLOY.md` (formats per family) and `docs/REMOTE.md` (one section per
    backend, with sign-in steps)
  - CHANGELOG, screenshots
- **Version:** 3.2.0, with the PyInstaller bundle and the CPU Docker image checked with
  the new extras, and nightly and CI green.

---

## Extras (pyproject)
| Extra | Packages |
| --- | --- |
| `kaggle` | kaggle |
| `modal` | modal |
| `export` | onnx, onnxruntime, executorch |
| `tracking` | mlflow, wandb |
| `hub` | huggingface_hub, datasets |
| `atari` | ale-py |
| `graph` (extended) | ogb |

## Verification (every phase)
- `ruff` + full `pytest` + `npm run build` + Playwright, locally and on CI.
- **Phase 0:** no CI job can run past its timeout. The end-to-end sweep passes for
  every sample and recipe. The 500-node design meets its time budget.
- **Phase 1:** an SSH run's curves and final metrics match a local run with the same
  seed, and it survives a dropped connection. The mocked Kaggle / Colab / Modal flows
  produce a complete run record. One real Kaggle kernel succeeds nightly.
- **Phase 2:** every exportable sample round-trips. Float exports match PyTorch
  within 1e-4. Int8 keeps top-1 within 1 point.
- **Phase 3:** a run mirrored to MLflow and W&B has the same metrics as the local
  record. A Hub push round-trips (push, then load the model and predict).
- **Phase 4:** each family's AutoML beats its baseline round on synthetic data.
- **Phase 5:** Glow beats a RealNVP of similar size in bits per dimension. A QAT text
  classifier stays within 1 point of float.

## Decisions (from review)
- **Remote training:** SSH, Kaggle, Colab and Modal, with SSH also covering GPU clouds
  that rent SSH machines. Colab is a notebook hand-off because Colab has no job API.
- **MLflow and W&B both ship.** They share one logger, so the second costs little.
- **Atari is in,** as an opt-in extra; users add ROMs themselves.
- **Bugs and CI come first** (Phase 0). Remote training follows as the biggest user ask,
  then export.
