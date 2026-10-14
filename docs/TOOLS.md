# Work with your tools

AI Made Easy mirrors runs to the experiment trackers you already use, registers models
in their registries, pushes to the Hugging Face Hub and reads datasets from Kaggle and
the Hub. Sign-in always stays with the tool itself: AI Made Easy never asks for, stores
or logs a password, token or key. Every integration is off until you turn it on.

```bash
pip install 'ai-made-easy[tracking]'   # MLflow + Weights & Biases
pip install 'ai-made-easy[hub]'        # the Hugging Face Hub and Hub datasets
pip install 'ai-made-easy[kaggle]'     # Kaggle datasets, competitions and kernels
```

## Experiment tracking: MLflow and Weights & Biases

Turn tracking on per project with **Run ▸ Experiment Tracking…** (the design's
`meta.tracking`), or for every run with `AIME_TRACKING=mlflow,wandb`. Each run then
mirrors as it goes:

- **at the start:** every block parameter, where it trains, and the design JSON
- **every epoch:** its metrics, with the epoch as the step
- **at the end:** the final metrics (as `final_<metric>` in MLflow, the summary in W&B),
  the run's result files and its status

Mirroring works the same on every backend (this computer, SSH, Kaggle, Colab, Modal,
Docker), because it follows the events the app already receives. A tracker that fails
(not signed in, server down) writes a line to the run's log and switches off. It never
stops the run. The run record lists where each run went (`tracking` in `run.json`).

| Tracker | Where runs go | Sign in |
| --- | --- | --- |
| MLflow | `MLFLOW_TRACKING_URI`; without it, a local SQLite store in `~/.aime/mlflow` (`mlflow ui --backend-store-uri sqlite:///$HOME/.aime/mlflow/mlflow.db`) | the server's own (`MLFLOW_TRACKING_USERNAME`, Databricks profiles, …) |
| W&B | the project named like your AI Made Easy project (or `WANDB_PROJECT`) | `wandb login`; `WANDB_MODE=offline` keeps runs local |

```bash
aime tracking                      # what's on, the MLflow store, what's installed
aime tracking project.json         # a project's setting
```

## Model registries

Registering or promoting a version in the app's model registry can also register it in
MLflow's model registry, with the stage as an alias (`models:/name@production`), or as a
W&B model artifact with the stage as an alias. Use **Experiments ▸ Models ▸ Share** or:

```bash
aime models register RUN_ID churn --also mlflow
aime models stage churn 3 production --also mlflow,wandb
```

An MLflow model opens as a design when the importers can read it: the PyTorch flavor
(a saved module) or an ONNX model.

```bash
aime import mlflow models:/churn/3 --shape 16 -o churn.json
```

Unpickling a saved PyTorch module runs code, so only open models you trust.

## The Hugging Face Hub

- **Push a run** as a private model repo: the trained files, the design, the run record
  and the model card AI Made Easy writes (with the Hub's `library_name` and tags).
- **Pull it back:** the download registers as a model version, so it packages and
  predicts like a local run.
- **Deploy a Space:** the FastAPI package becomes a Docker Space.

```bash
hf auth login                                   # once, the Hub's own sign-in
aime hub push RUN_ID you/churn-mlp              # --public to make it public
aime hub push churn/3 you/churn-mlp --model     # a registered version
aime hub pull you/churn-mlp                     # registers churn-mlp v1
aime hub space RUN_ID you/churn-api --formats onnx
```

The desktop app asks before anything leaves your computer.

## Datasets from Kaggle and the Hub

A source names where the data lives. It downloads once into `~/.aime/datasets` and is
then detected like a local file or folder:

| Source | What it downloads |
| --- | --- |
| `kaggle:owner/name` | a Kaggle dataset (add `/file.csv` for one file) |
| `kaggle-competition:name` | a competition's files; the training table is picked (`train.csv` before `test.csv`) |
| `hf:owner/repo` | a Hub dataset (add `:test` for another split): a Parquet table, or one folder per class for image datasets |

Use a source anywhere a data path goes: **File ▸ New from Task**, **Data ▸ Download a
Dataset**, `aime new <task> --data kaggle:owner/name`, or:

```bash
aime data fetch kaggle-competition:titanic
```

Competitions need you to join them and accept their rules on kaggle.com first.
Downloaded files are ordinary local data, so remote runs (SSH, Modal, …) take them along
like any other dataset.

**Submitting to a competition** asks every time, because submissions count against the
daily limit:

```bash
aime kaggle submit titanic predictions.csv -m "random forest, 5 folds"
```

## Run history for notebooks

```bash
aime runs export -o runs.parquet                # one row per run
aime runs export -o epochs.csv --epochs         # one row per epoch
```

Each run row has its identity, status, timing and backend, then `param.<block.param>`,
`final.<metric>` and `best.<metric>` columns. The desktop app has **Export ▸ Run
History**, and the web API has `GET /api/export/runs?format=parquet`.

## API, web and MCP

| | |
| --- | --- |
| REST | `GET/POST /api/tracking`, `POST /api/models/{name}/{version}/mirror`, `POST /api/hub/push`, `/api/hub/pull`, `/api/hub/space`, `POST /api/data/fetch`, `POST /api/kaggle/submit` (needs `"confirm": true`), `GET /api/export/runs` |
| MCP | `tracking_status`, `mirror_model`, `export_runs`, `fetch_dataset`, `open_mlflow_model`, `push_to_hub` |

Kaggle submissions aren't an MCP tool: they always go through a person.
