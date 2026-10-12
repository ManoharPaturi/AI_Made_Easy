"""Generative models: VAEs, GANs, diffusion, language models and sequence-to-sequence.

Importing the package registers the blocks, the tasks and their resolver, the codegen
helpers, the training loops (one per trainer kind), the design rules and the data
profiler.
"""
from ai_made_easy.core.generative import blocks as _blocks
from ai_made_easy.core.generative import helpers as _helpers

_helpers.register()
_blocks.register_all()

from ai_made_easy.core.generative import tasks  # noqa: E402, F401


def _register_trainers() -> None:
    from ai_made_easy.core.generative import image_template, sd_template, text_template
    from ai_made_easy.core.training.generate import register_trainer

    for kind in ("vae", "adversarial", "diffusion"):
        register_trainer(kind, image_template.render, needs_spec=False)
    register_trainer("text_to_image", sd_template.render, needs_spec=False)
    for kind in ("language_model", "seq2seq"):
        register_trainer(kind, text_template.render, needs_spec=False)


_register_trainers()


def _register_family() -> None:
    from ai_made_easy.core.families import Family, register_family
    from ai_made_easy.core.generative import sd_template
    from ai_made_easy.core.generative.rules import text_to_image_issues
    from ai_made_easy.core.targets import register_target

    def render(graph):  # noqa: ANN001, ANN202
        return sd_template.render(graph, None, "pytorch")

    register_family(Family(
        "text_to_image", "Stable Diffusion fine-tuning",
        "LoRA adapters that teach a pretrained Stable Diffusion a style or a subject "
        "(diffusers + peft).",
        detect=lambda types: "gen.sd_lora" in types, priority=30, frameworks=("pytorch",),
        targets=("sd_lora_train",), extras=("generative",), validate=text_to_image_issues,
        generators={"pytorch": render}))
    register_target("sd_lora_train", render, "Stable Diffusion LoRA script")


_register_family()
from ai_made_easy.core.generative import rules as _rules  # noqa: E402, F401
from ai_made_easy.core.generative import profile as _profile  # noqa: E402

_profile.register()
