# AI Made Easy

A visual, block-based AI model builder — drag blocks onto a canvas, wire them
into a model, **train it inside the app**, and **export clean PyTorch or Keras
code**. Think "Node-RED for AI".

> Status: **All 7 phases complete** + a UX restructure mined from Orange3,
> Langflow and Ryven — wrapped in the **Sunlit Classroom** look for young
> learners: warm light chrome flowing into a dusty graph-paper canvas, flat
> family-coloured blocks (model gold / activation coral / training lavender…),
> emoji palette tabs and big rounded controls. 123 blocks + LLM suite, architecture
> macros, ONNX/TorchScript export, in-app training with live curves, real
> datasets, model summary — and an **MCP server** exposing the whole engine
> to AI agents, plus an in-app AI assistant dock. See Agents & MCP.

## Quick start

```bash
git clone https://github.com/ManoharPaturi/AI_Made_Easy.git
cd AI_Made_Easy
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,dev]"     # add ",tensorflow" and/or ",llm" as needed
python -m ai_made_easy            # launch the GUI
```

Extras: `torch` (in-app training / ONNX / TorchScript), `tensorflow` (Keras
export), `llm` (generation / LoRA / RAG scripts), `mcp-server` (agent
tools), `dev` (tests). Pillow is required for the Image Folder dataset
block.

> **macOS + iCloud note:** if you keep the project inside an iCloud-synced
> folder (Desktop/Documents), put the venv outside it (e.g. `~/.aime-venv`)
> — iCloud sync can intermittently break Qt plugin discovery inside a
> synced venv. The app also auto-applies a Qt-plugins workaround at startup.

Headless (no GUI — the same path MCP agents will use later):

```bash
aime blocks                                      # list every block as JSON schemas
aime validate samples/mlp_mnist.json
aime gen samples/mlp_mnist.json -f pytorch -o exports   # model code
aime gen samples/mlp_mnist.json -f keras -o exports
aime train samples/mlp_mnist.json -f pytorch -o exports # runnable training script
aime train samples/mlp_mnist.json -f keras -o exports
python exports/mnist_mlp_train_pytorch.py        # actually trains (device auto)
aime onnx samples/mnist_cnn.json --run           # export model.onnx (+ checker)
aime jit samples/mnist_cnn.json --run            # export TorchScript (+ reload test)
aime llm samples/llm_rag_assistant.json          # LLM workflow script (gen/LoRA/RAG)
```

Run the tests:

```bash
pytest
```

## Agents & MCP

The same pure-Python core the UI drives is exposed as MCP tools — any MCP
client (Claude Desktop, ZCode, ...) can design, inspect, generate code for,
and **train models** through it:

```json
{
  "mcpServers": {
    "ai-made-easy": {
      "command": "/Users/you/.aime-venv/bin/aime-mcp",
      "args": []
    }
  }
}
```

Tools: `list_blocks`, `list_samples`, `read_sample`, `validate_graph`,
`generate_code` (pytorch/keras × model/train + llm), `summarize_model`,
`expand_architecture`, `start_training` → `run_id`, `get_run_status`,
`get_run_metrics`, `stop_run`. Graphs travel as JSON — the same files the
app saves and opens.

Headless CLI equivalents: `aime run project.json` (streams training events),
`aime summary project.json`, plus `gen/train/llm/onnx/jit`.

The in-app **AI Assistant dock** chats about the current graph with any
OpenAI-compatible endpoint and applies proposed graphs (validated first):
set `AIME_ASSISTANT_BASE_URL`, `AIME_ASSISTANT_API_KEY`,
`AIME_ASSISTANT_MODEL`.

## Architecture

See **docs/ARCHITECTURE.md** for the full structure. The short version:

```
ai_made_easy/
├── core/      # PURE PYTHON, zero Qt (enforced by test): registry + block
│              # library + graph IR + codegen + runners + summary + assistant
├── ui/
│   ├── app.py / context.py / workbench.py   # entry, composition root, shell
│   ├── stores.py / actions_catalog.py       # state + actions-as-data
│   ├── services/     # graph, process, export, project (all functionality)
│   ├── canvas/       # the ONLY OdenGraphQt package (enforced by test)
│   ├── features/     # header, palette, inspector, runconsole, controls
│   └── dialogs.py    # sample gallery, save-template, shortcuts sheet
├── worker/    # training worker executed in a subprocess
├── mcp/       # MCP server exposing core as tools
└── cli.py     # headless CLI over core
```

Three rules keep it scalable to agents/MCP later:

1. `core/` never imports Qt — every capability is a plain function over
   JSON-serializable data.
2. The graph **is** JSON — the file the UI saves is the artifact an agent edits.
3. The registry self-describes — `aime blocks` emits full JSON schemas.
4. The UI's structural rules (canvas boundary, Qt-free core, layout-only
   shell) are enforced by tests in `tests/test_structure.py`.

One IR, two code generators: the internal graph is canonical (channels-first,
explicit padding); all PyTorch-vs-Keras differences are resolved at codegen.

## Block library (112 built-ins + your custom blocks)

| Category | Count | Blocks |
|---|---|---|
| Data | 9 | Input, Output, Torchvision, CSV, Synthetic, NumPy (.npz), JSON/JSONL, Image Folder, HuggingFace |
| Preprocessing | 11 | Normalize, MinMaxScale, ToTensor, Split, Shuffle, DataLoader, Resize, CenterCrop, RandomFlip, RandomRotation, ColorJitter |
| Layers | 20 | Dense, Conv1/2/3D, ConvTranspose2/3D, Max/AvgPool, AdaptiveAvgPool2D, Global pools, LSTM, GRU, Embedding, Flatten |
| Activations | 15 | ReLU, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU, Mish, Tanh, Sigmoid, Softmax, LogSoftmax, Softplus, Hardswish, GLU |
| Attention | 2 | MultiheadAttention, TransformerEncoderLayer |
| Normalization | 6 | BatchNorm1/2D, LayerNorm, GroupNorm, Dropout, Dropout2D |
| Tensor Ops | 8 | Add, Multiply, Concatenate, Reshape, Permute, Squeeze, Unsqueeze, Lambda |
| Training | 26 | SGD, Adam, AdamW, Adagrad, RMSprop, Adadelta, Adamax, NAdam, RAdam; CE/MSE/BCE/L1/SmoothL1/Poisson/KLDiv/Focal losses; Step/Cosine/OneCycle/MultiStep/Exponential/WarmRestarts/Plateau/Linear schedulers; Trainer |
| Evaluation | 7 | Accuracy, Precision, Recall, F1, ConfusionMatrix, MSE, MAE |
| Architectures | 8 | MLP, CNN, VGG, ResNet-18, UNet, Autoencoder, LSTM Classifier, Transformer Classifier |
| LLM | 11 | HF Model, Tokenizer, Prompt Template, Generation Params, LoRA/QLoRA, SFT Dataset, Fine-tune, Embedding Model, Vector Store, Retriever, RAG Pipeline |

**LLM workflows** are canvas configs turned into standalone scripts by
`aime llm` / the ⤓ LLM Script button: text **generation**, **LoRA/QLoRA
fine-tuning** (peft, cosine schedule, adapter export), and **RAG**
(embeddings → numpy/faiss store → top-k retrieval → generation). Heavy
deps (`transformers peft datasets accelerate`) are optional — only the
generated scripts import them. All three verified end-to-end with real
models (tiny-gpt2, MiniLM).

**Architectures are macros**: drag one in, tweak its params (classes,
channels, depth...), then select it and hit **⤢ Expand** — it becomes
ordinary wired blocks (ResNet-18 = 65 nodes with real skip connections)
that you can edit, validate, and train. **💾 Save Selection** turns any
selected subgraph into a reusable block under the Custom palette category
(stored in `~/.aime/templates/`).

Model-flow blocks carry a `shape_fn` — the canvas computes shapes across
branches and merges live, flags mismatches, and feeds computed context
(`in_channels`, `in_features`, …) into codegen.

**Training-script export** collects the config blocks into a self-contained,
runnable script: dataset (torchvision/CSV/synthetic), preprocessing
(normalize/split/loader), optimizer, loss, scheduler, trainer (device auto:
MPS/CUDA/CPU, seed, early stopping), and metrics (accuracy, macro P/R/F1,
confusion matrix, MAE/MSE). No config blocks? Sensible defaults kick in
(synthetic data shaped like your model's IO, Adam, CrossEntropy).

## Roadmap

- **Phase 0** ✅ Spike: canvas → IR → valid PyTorch AND Keras code
- **Phase 1** ✅ 80-block library, DAG flow + merges, shape inference, live red-port validation, save/load
- **Phase 2** ✅ Training-script codegen (PyTorch + Keras), live code-preview pane, golden-file tests
- **Phase 3** ✅ Train in-app: worker subprocess + JSON event stream, live pyqtgraph curves, Stop button
- **Phase 4** ✅ Data breadth (NumPy/JSON/image-folder/HuggingFace), augmentation pipeline, 104-block catalog, analytic model-summary panel, sample gallery
- **Phase 5** ✅ Architecture macros + expand/splice, custom selection templates, ONNX + TorchScript export
- **Phase 6** ✅ LLM suite: HF model/tokenizer blocks, LoRA + QLoRA SFT script, generation script, RAG script (embed → store → retrieve → generate)
- **Phase 7** ✅ MCP server (12 tools, stdio), headless RunManager (`aime run`), IR-level architecture expansion, in-app AI assistant dock

## License

MIT
