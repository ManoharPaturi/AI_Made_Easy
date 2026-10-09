"""Generative tasks (VAE, GAN, diffusion, language modeling, sequence-to-sequence) and
the resolver that recognises them."""
from __future__ import annotations

from ai_made_easy.core.generative.blocks import GENERATIVE_DATA
from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

IMAGE_DATA = ("data.synthetic_images", "data.image_folder", "data.torchvision")
_IMAGE_METRICS = ["eval.fid", "eval.kid"]

register_task(Task(
    "vae_generation", "Variational autoencoder",
    "Learn a latent space that reconstructs images and generates new ones from the prior.",
    target="the input image", output_role="tensor", modalities=("image",), trainer_kind="vae",
    serving="images", default_metrics=("eval.fid",),
    meta={"loss_tasks": (), "losses": ["train.loss_mse", "train.loss_bce"],
          "metrics": _IMAGE_METRICS, "default_loss": None,
          "default_optimizer": "Adam (lr = 1e-3)"}))
register_task(Task(
    "gan_generation", "Image generation (GAN)",
    "Train a generator against a discriminator to produce realistic images from noise.",
    target="real images", output_role="tensor", modalities=("image",),
    trainer_kind="adversarial", serving="images", default_metrics=("eval.fid",),
    meta={"loss_tasks": (), "losses": [], "metrics": _IMAGE_METRICS, "default_loss": None,
          "default_optimizer": "Adam (lr = 2e-4, betas = 0.5, 0.999)"}))
register_task(Task(
    "diffusion_generation", "Image generation (diffusion)",
    "Learn to remove noise step by step, then generate images from pure noise.",
    target="noise", output_role="tensor", modalities=("image",), trainer_kind="diffusion",
    serving="images", default_metrics=("eval.fid",),
    meta={"loss_tasks": (), "losses": [], "metrics": _IMAGE_METRICS, "default_loss": None,
          "default_optimizer": "AdamW (lr = 2e-4)"}))
register_task(Task(
    "language_modeling", "Language modeling",
    "Predict the next token of text; generate continuations of a prompt.",
    target="next token", output_role="logits", modalities=("text",),
    trainer_kind="language_model", serving="text", default_metrics=("eval.perplexity",),
    meta={"loss_tasks": (), "losses": [], "metrics": ["eval.perplexity"], "default_loss": None,
          "default_optimizer": "AdamW (lr = 3e-4)"}))
register_task(Task(
    "sequence_to_sequence", "Sequence to sequence",
    "Map a source text to a target text (translation, rewriting, summarisation).",
    target="target text", output_role="logits", modalities=("text",), trainer_kind="seq2seq",
    serving="text", default_metrics=("eval.bleu", "eval.chrf"),
    meta={"loss_tasks": (), "losses": [], "default_loss": None,
          "metrics": ["eval.bleu", "eval.chrf", "eval.rouge_l"],
          "default_optimizer": "AdamW (lr = 5e-4)"}))

KIND_TASK = {"vae": "vae_generation", "adversarial": "gan_generation",
             "diffusion": "diffusion_generation", "language_model": "language_modeling",
             "seq2seq": "sequence_to_sequence"}


def _types(graph) -> set[str]:  # noqa: ANN001
    return {n.type_id for n in graph.nodes.values()}


def generative_kind(graph) -> str | None:  # noqa: ANN001
    """Trainer kind of a generative design, decided by its blocks."""
    types = _types(graph)
    if types & {"gen.noise_scheduler", "gen.diffusion_unet"}:
        return "diffusion"
    if "gen.discriminator" in types:
        return "adversarial"
    if "gen.reparameterize" in types:
        return "vae"
    data = {GENERATIVE_DATA.get(t) for t in types}
    if types & {"gen.seq2seq_transformer"} or "pairs" in data:
        return "seq2seq"
    if types & {"gen.gpt", "gen.causal_transformer"} or "text" in data:
        return "language_model"
    return None


def dataset_of(graph):  # noqa: ANN001, ANN201
    """The design's generative dataset block (image datasets include image folders and
    torchvision benchmarks)."""
    return next((n for n in graph.nodes.values()
                 if n.type_id in GENERATIVE_DATA or n.type_id in IMAGE_DATA), None)


def generative_task(graph) -> str | None:  # noqa: ANN001
    kind = generative_kind(graph)
    return KIND_TASK[kind] if kind else None


register_task_resolver(generative_task)
