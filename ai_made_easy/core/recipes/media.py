"""Recipes for images, text and audio clips sorted into one folder per class (or a text
table): CNNs, transfer learning, bag-of-embeddings, LSTM and transformer classifiers."""
from __future__ import annotations

from ai_made_easy.core.recipes import Context, Draft, Knob, Recipe, register_recipe
from ai_made_easy.core.recipes.tabular import CLASSIFY, EPOCHS, head, training

IMAGE_DEMO = ("fashion_mnist", 1, 28, 10)   # torchvision dataset, channels, size, classes
IMAGENET_MEAN, IMAGENET_STD = "0.485, 0.456, 0.406", "0.229, 0.224, 0.225"


def _lr(default: float) -> Knob:
    return Knob("lr", "float", default, 1e-5, 0.1, log=True, help="Optimizer learning rate")


def _classes(ctx: Context, demo: int) -> Context:
    """The context with the demo dataset's classes when there is no data."""
    if ctx.facts.demo and not ctx.facts.classes:
        ctx.facts.classes = [str(i) for i in range(demo)]
    return ctx


# ------------------------------------------------------------------ images

def _image_data(d: Draft, ctx: Context, size: int, *, imagenet: bool = False,
                augment: bool = False) -> tuple[int, int]:
    """Dataset + resize + augmentation + normalization; returns (channels, size)."""
    facts = ctx.facts
    if facts.demo:
        name, channels, native, _n = IMAGE_DEMO
        d.add("data.torchvision", f"the {name.replace('_', '-')} demo images (downloaded on "
                                  "first use): replace them with your own folder", nid="data",
              dataset=name)
        if imagenet:
            channels = 3
            d.add("prep.grayscale", "pretrained weights expect colour: copy the grey channel "
                                    "three times", nid="to_rgb", channels="3")
        if size != native:
            d.add("prep.resize", f"scales every image to {size}×{size}", nid="resize",
                  height=size, width=size)
    else:
        channels = 3
        d.add("data.image_folder", f"your images: one sub-folder per class "
                                   f"({len(facts.classes)} classes)", nid="data",
              root=facts.source)
        d.add("prep.resize", f"scales every image to {size}×{size} so they stack into "
                             "batches", nid="resize", height=size, width=size)
    if augment:
        d.add("prep.random_flip", "random mirror images: more variety from the same photos",
              nid="flip", mode="horizontal")
        d.add("prep.random_rotation", "small random rotations make the model robust to tilt",
              nid="rotate", degrees=10.0)
    if imagenet:
        d.add("prep.normalize", "the colour statistics the pretrained weights were trained "
                                "with", nid="normalize", mode="fixed", mean=IMAGENET_MEAN,
              std=IMAGENET_STD)
    else:
        mean = ", ".join(["0.5"] * channels)
        d.add("prep.normalize", "centres the pixel values around zero", nid="normalize",
              mode="fixed", mean=mean, std=mean)
    return channels, size


def _cnn(ctx: Context, *, augmented: bool) -> Draft:
    _classes(ctx, IMAGE_DEMO[3])
    name = "image_cnn_augmented" if augmented else "image_cnn"
    d = Draft(name, "Images — CNN" + (" with augmentation" if augmented else ""),
              "Convolutions learn edges, then shapes, then objects.")
    size = int(ctx["image_size"]) if not ctx.facts.demo else IMAGE_DEMO[2]
    channels, size = _image_data(d, ctx, size, augment=augmented)
    steps: list[tuple] = [("core.input", {"shape": f"{channels}, {size}, {size}"},
                           f"an image: {channels} channel(s) of {size}×{size} pixels")]
    width = int(ctx["width"])
    for stage in range(int(ctx["stages"])):
        steps += [("core.conv2d", {"out_channels": width * 2 ** stage, "kernel_size": 3,
                                   "padding": 1},
                   "3×3 filters find local patterns" if stage == 0 else
                   "more filters combine the earlier patterns into larger shapes"),
                  ("core.batch_norm2d", {}, "batch normalization keeps training stable"),
                  ("core.relu", {}, ""),
                  ("core.maxpool2d", {"kernel_size": 2, "stride": 2},
                   "halves the resolution: later filters see a wider area")]
    steps += [("core.global_avgpool2d", {}, "averages each filter over the image: one value "
                                            "per pattern, whatever the image size"),
              ("core.dropout", {"p": ctx["dropout"]}, "dropout against memorising images")]
    d.chain(*steps, *head(ctx), ("core.output", {}, ""))
    training(d, ctx, epochs=30, batch_size=64)
    if augmented:
        d.add("train.plateau_lr", "halves the learning rate when validation stops improving",
              nid="scheduler", factor=0.5, patience=3)
    return d


def _transfer(ctx: Context) -> Draft:
    _classes(ctx, IMAGE_DEMO[3])
    d = Draft("image_transfer", "Images — transfer learning",
              "A network pretrained on ImageNet with a new classifier on top.")
    channels, size = _image_data(d, ctx, int(ctx["image_size"]), imagenet=True, augment=True)
    d.chain(("core.input", {"shape": f"{channels}, {size}, {size}"},
             f"an RGB image of {size}×{size} pixels"),
            ("core.pretrained_backbone", {"architecture": ctx["architecture"],
                                          "weights": "imagenet", "freeze": True},
             f"{ctx['architecture']} pretrained on ImageNet's 1.2M photos: its features "
             "already recognise textures, parts and objects; frozen, so only the new head "
             "learns"),
            ("core.dropout", {"p": 0.2}, "dropout on the features against overfitting"),
            *head(ctx), ("core.output", {}, ""))
    training(d, ctx, optimizer="train.adam", epochs=8, batch_size=32)
    return d


def _vit(ctx: Context) -> Draft:
    _classes(ctx, IMAGE_DEMO[3])
    d = Draft("image_vit", "Images — Vision Transformer",
              "Patches of the image become tokens; attention compares every patch with "
              "every other.")
    size = int(ctx["image_size"]) if not ctx.facts.demo else IMAGE_DEMO[2]
    channels, size = _image_data(d, ctx, size, augment=True)
    patch = int(ctx["patch"])
    if size % patch:
        patch = next(p for p in (4, 7, 2, 1) if size % p == 0)
    dim, tokens = int(ctx["width"]), (size // patch) ** 2
    steps: list[tuple] = [
        ("core.input", {"shape": f"{channels}, {size}, {size}"},
         f"an image: {channels} channel(s) of {size}×{size} pixels"),
        ("vision.patch_embedding", {"patch": patch, "dim": dim, "cls_token": False},
         f"cuts the image into {tokens} patches of {patch}×{patch} pixels and embeds each "
         f"as a {dim}-number token, with a learned position")]
    for i in range(int(ctx["depth"])):
        steps.append(("core.transformer_encoder",
                      {"nhead": 4, "dim_feedforward": 2 * dim, "dropout": 0.1,
                       "activation": "gelu", "norm_first": True},
                      "self-attention lets every patch look at every other" if i == 0 else ""))
    steps += [("core.layer_norm", {}, "a final normalization, as in ViT"),
              ("core.reduce", {"op": "mean", "dim": 0},
               "averages the patch tokens into one image summary")]
    d.chain(*steps, *head(ctx), ("core.output", {}, ""))
    training(d, ctx, epochs=40, batch_size=64)
    d.add("train.warmup_cosine_lr", "warms the learning rate up, then decays it: transformers "
                                 "train poorly without warmup", nid="scheduler")
    return d


def _convnext(ctx: Context) -> Draft:
    _classes(ctx, IMAGE_DEMO[3])
    d = Draft("image_convnext", "Images — ConvNeXt",
              "A modern CNN: large depthwise filters, LayerNorm and an inverted MLP per "
              "block.")
    size = int(ctx["image_size"]) if not ctx.facts.demo else IMAGE_DEMO[2]
    channels, size = _image_data(d, ctx, size, augment=True)
    width = int(ctx["width"])
    steps: list[tuple] = [
        ("core.input", {"shape": f"{channels}, {size}, {size}"},
         f"an image: {channels} channel(s) of {size}×{size} pixels"),
        ("core.conv2d", {"out_channels": width, "kernel_size": 2, "stride": 2},
         "a 2×2 patchify stem halves the resolution")]
    for stage in range(int(ctx["stages"])):
        if stage:
            steps.append(("core.conv2d", {"out_channels": width * 2 ** stage, "kernel_size": 2,
                                          "stride": 2},
                          "downsamples between stages and widens the features"))
        steps.append(("vision.convnext_block", {"drop_path": 0.1},
                      "7×7 depthwise conv, LayerNorm and an inverted MLP, with a residual"
                      if stage == 0 else ""))
    steps += [("core.global_avgpool2d", {}, "one value per feature over the image")]
    d.chain(*steps, *head(ctx), ("core.output", {}, ""))
    training(d, ctx, epochs=30, batch_size=64)
    return d


def _arcface(ctx: Context) -> Draft:
    _classes(ctx, IMAGE_DEMO[3])
    d = Draft("image_arcface", "Images — ArcFace embeddings",
              "A CNN that maps each image to an embedding; ArcFace pulls each class around "
              "its own direction with a margin between classes.")
    size = int(ctx["image_size"]) if not ctx.facts.demo else IMAGE_DEMO[2]
    channels, size = _image_data(d, ctx, size, augment=True)
    width = int(ctx["width"])
    steps: list[tuple] = [("core.input", {"shape": f"{channels}, {size}, {size}"},
                           f"an image: {channels} channel(s) of {size}×{size} pixels")]
    for stage in range(int(ctx["stages"])):
        steps += [("core.conv2d", {"out_channels": width * 2 ** stage, "kernel_size": 3,
                                   "padding": 1},
                   "3×3 filters find local patterns" if stage == 0 else ""),
                  ("core.batch_norm2d", {}, ""),
                  ("core.relu", {}, ""),
                  ("core.maxpool2d", {"kernel_size": 2, "stride": 2}, "")]
    n = max(len(ctx.facts.classes or []), 2)
    steps += [("core.global_avgpool2d", {}, "one value per filter"),
              ("core.dense", {"units": int(ctx["embedding"])},
               f"the embedding: {int(ctx['embedding'])} numbers per image, compared by angle")]
    d.chain(*steps, ("core.output", {}, ""))
    d.add("train.loss_arcface", f"ArcFace: {n} class centres on the sphere, with an angular "
                                "margin, so same-class images cluster tightly", nid="loss",
          classes=n)
    d.add("train.adamw", "AdamW: Adam with decoupled weight decay", nid="optimizer",
          lr=ctx["lr"])
    d.add("train.trainer", "early stopping ends training when the validation loss stops "
                           "improving", nid="trainer", epochs=ctx["epochs"] or 30,
          batch_size=128, early_stopping_patience=5)
    d.add("eval.accuracy", "accuracy of the nearest class centre", nid="accuracy")
    return d


def _simclr(ctx: Context) -> Draft:
    _classes(ctx, IMAGE_DEMO[3])
    d = Draft("image_simclr", "Images — SimCLR (self-supervised)",
              "Learns image features without labels: two random augmentations of the same "
              "image should land close together, and far from every other image.")
    size = int(ctx["image_size"]) if not ctx.facts.demo else IMAGE_DEMO[2]
    channels, size = _image_data(d, ctx, size, augment=True)
    width = int(ctx["width"])
    steps: list[tuple] = [("core.input", {"shape": f"{channels}, {size}, {size}"},
                           f"an image: {channels} channel(s) of {size}×{size} pixels")]
    for stage in range(int(ctx["stages"])):
        steps += [("core.conv2d", {"out_channels": width * 2 ** stage, "kernel_size": 3,
                                   "padding": 1},
                   "the encoder: its features are what you keep" if stage == 0 else ""),
                  ("core.batch_norm2d", {}, ""),
                  ("core.relu", {}, ""),
                  ("core.maxpool2d", {"kernel_size": 2, "stride": 2}, "")]
    n = max(len(ctx.facts.classes or []), 2)
    projection = int(ctx["projection"])
    steps += [("core.global_avgpool2d", {}, "one value per filter: the image's features"),
              ("core.dense", {"units": projection * 2},
               "a projection head: the contrastive loss works on its output"),
              ("core.relu", {}, ""),
              ("core.dense", {"units": projection}, f"a {projection}-number projection")]
    d.chain(*steps, ("core.output", {}, ""))
    d.add("train.loss_nt_xent", "NT-Xent: each image is augmented twice and must pick out "
                                "its twin among the batch. Labels only measure the result",
          nid="loss", classes=n, temperature=0.5)
    d.add("train.adamw", "AdamW: Adam with decoupled weight decay", nid="optimizer",
          lr=ctx["lr"])
    d.add("train.trainer", "large batches give the loss more images to tell apart",
          nid="trainer", epochs=ctx["epochs"] or 30, batch_size=256,
          early_stopping_patience=8)
    d.add("eval.accuracy", "accuracy of the nearest class average: how well the features "
                           "separate classes they never saw labels for", nid="accuracy")
    return d


IMAGE_KINDS = ("demo", "image_folder")
register_recipe(Recipe(
    "image_cnn", "Small CNN", CLASSIFY, "small", "image",
    "Two convolution stages and a linear classifier: trains in minutes on a CPU.",
    lambda ctx: _cnn(ctx, augmented=False), data_kinds=IMAGE_KINDS,
    demo_tasks=("multiclass",), rows=(0, 50_000),
    downloads="Fashion-MNIST (30 MB) for the demo",
    knobs=(_lr(1e-3), Knob("width", "int", 32, 8, 256, help="Filters in the first stage"),
           Knob("stages", "int", 2, 1, 5, help="Convolution stages"),
           Knob("dropout", "float", 0.2, 0.0, 0.6),
           Knob("image_size", "choice", 64, values=(32, 64, 96, 128)), EPOCHS(20)),
    strengths="small, fast and easy to inspect"))
register_recipe(Recipe(
    "image_cnn_augmented", "CNN + augmentation", CLASSIFY, "medium", "image",
    "A deeper CNN with flips, rotations and a learning-rate schedule.",
    lambda ctx: _cnn(ctx, augmented=True), data_kinds=IMAGE_KINDS, demo_tasks=("multiclass",),
    rows=(1_000, 10**7),
    downloads="Fashion-MNIST (30 MB) for the demo",
    knobs=(_lr(1e-3), Knob("width", "int", 32, 8, 256, help="Filters in the first stage"),
           Knob("stages", "int", 3, 1, 5, help="Convolution stages"),
           Knob("dropout", "float", 0.3, 0.0, 0.6),
           Knob("image_size", "choice", 64, values=(32, 64, 96, 128)), EPOCHS(30)),
    strengths="augmentation gets more out of every image"))
register_recipe(Recipe(
    "image_transfer", "Transfer learning (ResNet)", CLASSIFY, "pretrained", "image",
    "An ImageNet-pretrained ResNet with a new classification head.", _transfer,
    data_kinds=IMAGE_KINDS, demo_tasks=("multiclass",), rows=(0, 10**6), priority=0.6,
    downloads="ImageNet weights (45-100 MB)",
    knobs=(_lr(1e-3), Knob("architecture", "choice", "resnet18",
                           values=("resnet18", "resnet34", "resnet50")),
           Knob("image_size", "choice", 128, values=(64, 96, 128, 160, 224)), EPOCHS(8)),
    strengths="the best accuracy from few images: it starts from learned features"))
register_recipe(Recipe(
    "image_vit", "Vision Transformer (from scratch)", CLASSIFY, "medium", "image",
    "Patch embeddings and transformer encoder layers, trained from scratch.", _vit,
    data_kinds=IMAGE_KINDS, demo_tasks=("multiclass",), rows=(5_000, 10**7), priority=-0.2,
    downloads="Fashion-MNIST (30 MB) for the demo",
    knobs=(_lr(5e-4), Knob("width", "int", 64, 16, 512, help="Token width"),
           Knob("depth", "int", 4, 1, 12, help="Transformer layers"),
           Knob("patch", "choice", 4, values=(2, 4, 8, 16)),
           Knob("image_size", "choice", 32, values=(32, 64, 96, 128)), EPOCHS(40)),
    strengths="scales with data; attention sees the whole image from the first layer"))
register_recipe(Recipe(
    "image_convnext", "ConvNeXt (from scratch)", CLASSIFY, "medium", "image",
    "A modern convolutional network: depthwise 7×7 filters, LayerNorm, GELU MLPs.",
    _convnext, data_kinds=IMAGE_KINDS, demo_tasks=("multiclass",), rows=(2_000, 10**7),
    priority=-0.1, downloads="Fashion-MNIST (30 MB) for the demo",
    knobs=(_lr(1e-3), Knob("width", "int", 32, 8, 256, help="Features in the first stage"),
           Knob("stages", "int", 3, 1, 4, help="Stages (each halves the resolution)"),
           Knob("image_size", "choice", 64, values=(32, 64, 96, 128)), EPOCHS(30)),
    strengths="CNN speed with transformer-era accuracy"))
register_recipe(Recipe(
    "image_arcface", "ArcFace embeddings", CLASSIFY, "medium", "image",
    "Learns an embedding per image (face / product / item recognition): new images are "
    "matched by angle, and new classes can be added without retraining.",
    _arcface, data_kinds=IMAGE_KINDS, demo_tasks=("multiclass",), rows=(2_000, 10**7),
    priority=-0.2, downloads="Fashion-MNIST (30 MB) for the demo",
    knobs=(_lr(1e-3), Knob("width", "int", 32, 8, 256, help="Filters in the first stage"),
           Knob("stages", "int", 3, 1, 5, help="Convolution stages"),
           Knob("embedding", "int", 128, 16, 1024, help="Embedding size"),
           Knob("image_size", "choice", 64, values=(32, 64, 96, 128)), EPOCHS(30)),
    strengths="embeddings for retrieval and verification, not only a class"))
register_recipe(Recipe(
    "image_simclr", "SimCLR (self-supervised)", CLASSIFY, "large", "image",
    "Pretrains an encoder without using the labels, by contrasting two augmentations of "
    "each image; fine-tune it afterwards with a small labelled set.",
    _simclr, data_kinds=IMAGE_KINDS, demo_tasks=("multiclass",), rows=(5_000, 10**8),
    priority=-0.3, downloads="Fashion-MNIST (30 MB) for the demo",
    knobs=(_lr(1e-3), Knob("width", "int", 32, 8, 256, help="Filters in the first stage"),
           Knob("stages", "int", 3, 1, 5, help="Convolution stages"),
           Knob("projection", "int", 64, 16, 512, help="Projection size"),
           Knob("image_size", "choice", 64, values=(32, 64, 96, 128)), EPOCHS(30)),
    strengths="uses unlabelled images; features that transfer"))


# ------------------------------------------------------------------ text

VOCAB, MAX_LENGTH = 20_000, 128


def _text_data(d: Draft, ctx: Context) -> None:
    facts = ctx.facts
    if facts.kind == "text_table":
        d.add("data.text_csv", f"your table: '{facts.text_column}' holds the text, "
                               f"'{facts.target}' the label", nid="data", path=facts.source,
              format=facts.format or "csv", text_column=facts.text_column,
              label_column=facts.target)
    else:
        d.add("data.text_folder", "your documents: one sub-folder per class", nid="data",
              root=facts.source)
    d.add("prep.text_clean", "lower-cases the text and strips markup", nid="clean")
    d.add("prep.tokenize", f"splits the text into words and keeps the {VOCAB:,} most "
                           f"frequent; each document becomes {MAX_LENGTH} word ids",
          nid="tokenize", method="word", max_length=MAX_LENGTH, vocab_size=VOCAB)


def _text(ctx: Context, kind: str) -> Draft:
    d = Draft(f"text_{kind}", f"Text — {kind}", "Word embeddings and a classifier.")
    _text_data(d, ctx)
    dim = int(ctx["dim"])
    steps: list[tuple] = [
        ("core.input", {"shape": str(MAX_LENGTH), "dtype": "int64"},
         f"a document as {MAX_LENGTH} word ids"),
        ("core.embedding", {"num_embeddings": VOCAB + 2, "embedding_dim": dim},
         f"learns a {dim}-number vector for every word")]
    if kind == "lstm":
        steps.append(("core.lstm", {"hidden_size": dim, "bidirectional": True,
                                    "return_sequences": True},
                      "a bidirectional LSTM reads the words in order, both ways, so word "
                      "order and negation count"))
    elif kind == "transformer":
        steps += [("core.positional_encoding", {}, "tells attention where each word is"),
                  *[("core.transformer_encoder", {"nhead": 4, "dim_feedforward": dim * 4},
                     "self-attention: every word looks at every other word")
                    for _ in range(int(ctx["layers"]))]]
    steps += [("core.mean_over_time", {}, "averages over the words: one vector per document"),
              ("core.dropout", {"p": 0.3}, "dropout against memorising documents")]
    d.chain(*steps, *head(ctx), ("core.output", {}, ""))
    training(d, ctx, epochs=15, batch_size=64)
    return d


TEXT_KINDS = ("text_folder", "text_table")
register_recipe(Recipe(
    "text_bag_of_embeddings", "Bag of embeddings", CLASSIFY, "small", "text",
    "Averages learned word vectors: fast, and strong when the words themselves decide.",
    lambda ctx: _text(ctx, "bag"), data_kinds=TEXT_KINDS, rows=(0, 10**7),
    knobs=(_lr(3e-3), Knob("dim", "choice", 64, values=(32, 64, 128, 256),
                           help="Embedding size"), EPOCHS(15)),
    strengths="trains in seconds; a strong baseline for topic and sentiment"))
register_recipe(Recipe(
    "text_lstm", "Bidirectional LSTM", CLASSIFY, "medium", "text",
    "Reads the words in order with a bidirectional LSTM.",
    lambda ctx: _text(ctx, "lstm"), data_kinds=TEXT_KINDS, rows=(2_000, 10**7),
    knobs=(_lr(2e-3), Knob("dim", "choice", 64, values=(32, 64, 128, 256),
                           help="Embedding and LSTM size"), EPOCHS(15)),
    strengths="word order matters (negation, phrasing)"))
register_recipe(Recipe(
    "text_transformer", "Transformer encoder", CLASSIFY, "large", "text",
    "Self-attention layers over the words, trained from scratch.",
    lambda ctx: _text(ctx, "transformer"), data_kinds=TEXT_KINDS, rows=(20_000, 10**8),
    knobs=(_lr(5e-4), Knob("dim", "choice", 64, values=(64, 128, 256), help="Model width"),
           Knob("layers", "int", 2, 1, 6, help="Transformer layers"), EPOCHS(15)),
    strengths="captures long-range context with enough data"))


# ------------------------------------------------------------------ audio

def _audio(ctx: Context) -> Draft:
    facts = ctx.facts
    d = Draft("audio_mel_cnn", "Audio — mel-spectrogram CNN",
              "A CNN over the mel spectrogram of each clip.")
    d.add("data.audio_folder", f"your clips: one sub-folder per class ({len(facts.classes)} "
                               "classes), resampled to 16 kHz and cut to 1 s", nid="data",
          root=facts.source, sample_rate=16000, duration=1.0)
    steps: list[tuple] = [
        ("core.input", {"shape": "1, 16000"}, "one second of audio at 16 kHz"),
        ("audio.mel_spectrogram", {"sample_rate": 16000, "n_fft": 400, "hop_length": 160,
                                   "n_mels": 40},
         "turns the waveform into a picture of which frequencies sound when, on the scale "
         "human hearing uses")]
    width = int(ctx["width"])
    for stage in range(3):
        steps += [("core.conv2d", {"out_channels": width * 2 ** stage},
                   "filters find patterns in time and frequency"),
                  ("core.batch_norm2d", {}, ""), ("core.relu", {}, ""),
                  ("core.maxpool2d", {}, "")]
    steps.append(("core.global_avgpool2d", {}, "one value per pattern for the whole clip"))
    d.chain(*steps, *head(ctx), ("core.output", {}, ""))
    training(d, ctx, epochs=25, batch_size=32)
    return d


register_recipe(Recipe(
    "audio_mel_cnn", "Mel-spectrogram CNN", CLASSIFY, "small", "audio",
    "Treats the spectrogram as an image: the standard recipe for sound classification.",
    _audio, data_kinds=("audio_folder",),
    knobs=(_lr(1e-3), Knob("width", "int", 16, 8, 128, help="Filters in the first stage"),
           EPOCHS(25)),
    strengths="works for commands, events and other short sounds"))
