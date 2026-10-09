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
    from ai_made_easy.core.generative import image_template, text_template
    from ai_made_easy.core.training.generate import register_trainer

    for kind in ("vae", "adversarial", "diffusion"):
        register_trainer(kind, image_template.render, needs_spec=False)
    for kind in ("language_model", "seq2seq"):
        register_trainer(kind, text_template.render, needs_spec=False)


_register_trainers()
from ai_made_easy.core.generative import rules as _rules  # noqa: E402, F401
from ai_made_easy.core.generative import profile as _profile  # noqa: E402

_profile.register()
