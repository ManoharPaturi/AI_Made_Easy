"""Pretrained recipes: classifiers on the Hugging Face, timm and DINOv2 encoder blocks.

Each modality comes in two variants. A **linear probe** freezes the encoder and trains
only a new head: fast, little memory, and the right choice for a few hundred examples.
**Fine-tuning** trains the whole encoder at a small learning rate: more accurate with
more data, at the cost of memory and time. The wizard shows what each one downloads
(size and licence, from ``core.zoo``) and hides them when they do not fit the budget.
"""
from __future__ import annotations

from ai_made_easy.core.recipes import Context, Draft, Knob, Recipe, register_recipe
from ai_made_easy.core.recipes.media import IMAGE_DEMO, _classes
from ai_made_easy.core.recipes.tabular import CLASSIFY, EPOCHS, head, training
from ai_made_easy.core.zoo import catalog

TEXT_MODELS = ("distilbert-base-uncased", "sentence-transformers/all-MiniLM-L6-v2",
               "google/electra-small-discriminator", "bert-base-uncased", "roberta-base",
               "microsoft/deberta-v3-small")
AUDIO_MODELS = ("facebook/wav2vec2-base", "facebook/hubert-base-ls960",
                "microsoft/wavlm-base-plus", "MIT/ast-finetuned-audioset-10-10-0.4593")
TIMM_MODELS = ("convnext_atto", "efficientnet_b0", "mobilenetv4_conv_small", "resnet50d",
               "convnext_tiny", "vit_small_patch16_224", "eva02_small_patch14_224")
DINO_MODELS = ("facebook/dinov2-small", "facebook/dinov2-base")
SPEECH_DEMO = {"task": "keywords", "n_clips": 600, "n_keywords": 6, "noise": 0.03,
               "sample_rate": 16000, "duration": 1.0}
TEXT_KINDS = ("text_folder", "text_table")


def _lr(default: float) -> Knob:
    return Knob("lr", "float", default, 1e-6, 0.1, log=True, help="Optimizer learning rate")


def _model(name: str, default: str, values: tuple[str, ...], what: str) -> Knob:
    return Knob(name, "choice", default, values=values, help=what)


def _encoder_note(model: str, finetune: bool) -> str:
    row = next((r for r in catalog("hf_encoders") if r["name"] == model), None)
    size = f" ({row['params_m']:g}M parameters)" if row else ""
    return (f"{model}{size}, pretrained on unlabelled data: " +
            ("every layer keeps learning at a small learning rate (fine-tuning)" if finetune
             else "frozen, so only the new head learns (linear probe)"))


def _probe_or_tune(d: Draft, ctx: Context, finetune: bool, encoder: list[tuple], *,
                   epochs: tuple[int, int], batch: tuple[int, int]) -> None:
    """The input and encoder steps, then dropout, the task's head and the training setup."""
    d.chain(*encoder, ("core.dropout", {"p": 0.1 if finetune else 0.2},
                       "dropout on the features against overfitting"), *head(ctx),
            ("core.output", {}, ""))
    training(d, ctx, optimizer="train.adamw", epochs=epochs[finetune],
             batch_size=batch[finetune])


# ------------------------------------------------------------------ text

def _text(ctx: Context, finetune: bool) -> Draft:
    facts, model, length = ctx.facts, ctx["model"], int(ctx["max_length"])
    d = Draft(f"text_{'finetune' if finetune else 'probe'}",
              f"Text — {model.split('/')[-1]} {'fine-tuned' if finetune else 'probe'}",
              "A pretrained transformer encoder with a new classification head.")
    if facts.kind == "text_table":
        d.add("data.text_csv", f"your table: '{facts.text_column}' holds the text, "
                               f"'{facts.target}' the label", nid="data", path=facts.source,
              format=facts.format or "csv", text_column=facts.text_column,
              label_column=facts.target)
    else:
        d.add("data.text_folder", "your documents: one sub-folder per class", nid="data",
              root=facts.source)
    d.add("prep.tokenize", f"the encoder's own tokenizer: sub-word pieces it was pretrained "
                           f"on, padded or cut to {length} tokens", nid="tokenize",
          method="huggingface", hf_tokenizer=model, max_length=length)
    _probe_or_tune(d, ctx, finetune, [
        ("core.input", {"shape": str(length), "dtype": "int64"},
         f"a document as {length} token ids"),
        ("core.hf_text_encoder", {"model_id": model, "pooling": "mean", "freeze": not finetune,
                                  "weights": "pretrained"}, _encoder_note(model, finetune))],
        epochs=(15, 4), batch=(32, 16))
    return d


# ------------------------------------------------------------------ audio

def _audio(ctx: Context, finetune: bool) -> Draft:
    facts, model = ctx.facts, ctx["model"]
    d = Draft(f"audio_{'finetune' if finetune else 'probe'}",
              f"Audio — {model.split('/')[-1]} {'fine-tuned' if finetune else 'probe'}",
              "A self-supervised speech encoder with a new classification head.")
    if facts.demo:
        d.add("data.synthetic_speech", "generated spoken-keyword clips: replace them with a "
                                       "folder of your own recordings", nid="data",
              **SPEECH_DEMO)
    else:
        d.add("data.audio_folder", f"your clips: one sub-folder per class "
                                   f"({len(facts.classes)} classes), resampled to 16 kHz and "
                                   "cut to 1 s", nid="data", root=facts.source,
              sample_rate=16000, duration=1.0)
    _probe_or_tune(d, ctx, finetune, [
        ("core.input", {"shape": "1, 16000"}, "one second of audio at 16 kHz, the rate the "
                                              "encoder was pretrained on"),
        ("audio.hf_encoder", {"model_id": model, "weights": "pretrained",
                              "freeze": not finetune, "pooling": "mean"},
         _encoder_note(model, finetune) + "; the frames are averaged into one vector")],
        epochs=(20, 6), batch=(32, 8))
    return d


# ------------------------------------------------------------------ images

def _image(ctx: Context, finetune: bool, dino: bool) -> Draft:
    _classes(ctx, IMAGE_DEMO[3])
    facts, model, size = ctx.facts, ctx["model"], int(ctx["image_size"])
    name = model.split("/")[-1]
    d = Draft(f"{'dinov2' if dino else 'timm'}_{'finetune' if finetune else 'probe'}",
              f"Images — {name} {'fine-tuned' if finetune else 'probe'}",
              "A pretrained image encoder with a new classification head.")
    if facts.demo:
        dataset, channels, native, _n = IMAGE_DEMO
        d.add("data.torchvision", f"the {dataset.replace('_', '-')} demo images (downloaded "
                                  "on first use): replace them with your own folder",
              nid="data", dataset=dataset)
    else:
        channels = 3
        d.add("data.image_folder", f"your images: one sub-folder per class "
                                   f"({len(facts.classes)} classes)", nid="data",
              root=facts.source)
    d.add("prep.resize", f"scales every image to {size}×{size}", nid="resize", height=size,
          width=size)
    if finetune:
        d.add("prep.random_flip", "random mirror images: more variety from the same photos",
              nid="flip", mode="horizontal")
    block = ("vision.hf_image_encoder", {"model_id": model, "weights": "pretrained",
                                         "freeze": not finetune}) if dino else \
        ("vision.timm_backbone", {"model": model, "weights": "imagenet", "freeze": not finetune})
    what = ("DINOv2: self-supervised on 142M images, strong features without labels"
            if dino else f"{model} from timm, pretrained on ImageNet")
    how = ("; every layer keeps learning at a small learning rate" if finetune
           else "; frozen, so only the new head learns")
    _probe_or_tune(d, ctx, finetune, [
        ("core.input", {"shape": f"{channels}, {size}, {size}"},
         f"an image of {size}×{size} pixels (the encoder applies the ImageNet colour "
         "statistics itself)"),
        (block[0], block[1], what + how)], epochs=(15, 6), batch=(64, 32))
    return d


# ------------------------------------------------------------------ registry

TEXT = dict(data_kinds=TEXT_KINDS, requires=("transformers",), extra="text")
AUDIO = dict(data_kinds=("demo", "audio_folder"), demo_tasks=("keyword_spotting",),
             requires=("transformers",), extra="audio")
TIMM = dict(data_kinds=("demo", "image_folder"), demo_tasks=("multiclass",),
            requires=("timm",), extra="vision-tasks")
DINO = dict(TIMM, requires=("transformers",))
LENGTH = Knob("max_length", "choice", 128, values=(64, 128, 256, 512),
              help="Tokens per document")

for finetune in (False, True):
    tier, probe = "pretrained", not finetune
    register_recipe(Recipe(
        f"text_encoder_{'finetune' if finetune else 'probe'}",
        "Pretrained text encoder — " + ("fine-tune" if finetune else "linear probe"),
        CLASSIFY, tier, "text",
        "A Hugging Face transformer encoder (DistilBERT by default) with a new head; " +
        ("all of it trains" if finetune else "the encoder stays frozen") + ".",
        lambda ctx, f=finetune: _text(ctx, f), **TEXT,
        rows=(2_000, 10**8) if finetune else (0, 10**6), priority=0.8 if probe else 0.7,
        knobs=(_lr(3e-5 if finetune else 1e-3),
               _model("model", "distilbert-base-uncased", TEXT_MODELS, "Encoder checkpoint"),
               LENGTH, EPOCHS(4 if finetune else 15)),
        strengths="the most accurate text classifier with enough examples" if finetune
        else "strong accuracy from a few hundred labelled documents"))
    register_recipe(Recipe(
        f"audio_encoder_{'finetune' if finetune else 'probe'}",
        "Pretrained speech encoder — " + ("fine-tune" if finetune else "linear probe"),
        ("keyword_spotting", *CLASSIFY), tier, "audio",
        "wav2vec 2.0 / HuBERT / WavLM with a new head; " +
        ("all of it trains" if finetune else "the encoder stays frozen") + ".",
        lambda ctx, f=finetune: _audio(ctx, f), **AUDIO,
        rows=(1_000, 10**7) if finetune else (0, 10**6), priority=0.8 if probe else 0.7,
        knobs=(_lr(3e-5 if finetune else 1e-3),
               _model("model", "facebook/wav2vec2-base", AUDIO_MODELS, "Encoder checkpoint"),
               EPOCHS(6 if finetune else 20)),
        strengths="robust to noise and accents: learned from hundreds of hours of speech"))
    register_recipe(Recipe(
        f"timm_{'finetune' if finetune else 'probe'}",
        "timm image model — " + ("fine-tune" if finetune else "linear probe"),
        CLASSIFY, tier, "image",
        "A timm ImageNet model (ConvNeXt, EfficientNet, ViT, ...) with a new head; " +
        ("all of it trains" if finetune else "the backbone stays frozen") + ".",
        lambda ctx, f=finetune: _image(ctx, f, dino=False), **TIMM,
        rows=(2_000, 10**8) if finetune else (0, 10**6), priority=0.7 if probe else 0.6,
        knobs=(_lr(1e-4 if finetune else 1e-3),
               _model("model", "convnext_atto", TIMM_MODELS, "Backbone"),
               Knob("image_size", "choice", 128, values=(64, 96, 128, 160, 224)),
               EPOCHS(6 if finetune else 15)),
        strengths="modern architectures, small to large" if probe
        else "the best accuracy when you have thousands of images"))
    register_recipe(Recipe(
        f"dinov2_{'finetune' if finetune else 'probe'}",
        "DINOv2 — " + ("fine-tune" if finetune else "linear probe"),
        CLASSIFY, tier, "image",
        "Meta's self-supervised vision transformer with a new head; " +
        ("all of it trains" if finetune else "the encoder stays frozen") + ".",
        lambda ctx, f=finetune: _image(ctx, f, dino=True), **DINO,
        rows=(2_000, 10**8) if finetune else (0, 10**6), priority=0.75 if probe else 0.6,
        knobs=(_lr(2e-5 if finetune else 1e-3),
               _model("model", "facebook/dinov2-small", DINO_MODELS, "Encoder checkpoint"),
               Knob("image_size", "choice", 112, values=(112, 168, 224)),
               EPOCHS(6 if finetune else 15)),
        strengths="general-purpose features: excellent linear probes, even far from "
                  "ImageNet (medical, satellite, products)"))
