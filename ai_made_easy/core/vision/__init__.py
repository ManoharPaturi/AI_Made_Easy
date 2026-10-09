"""Vision tasks: object detection, instance / semantic segmentation and keypoints.

Importing the package registers the blocks (``blocks``), the tasks and their
resolver (``tasks``), the design rules (``rules``), the codegen helpers
(``helpers``) and the training loops (``template``).
"""
from ai_made_easy.core.vision import helpers as _helpers
from ai_made_easy.core.vision import blocks as _blocks

_helpers.register()
_blocks.register_all()

from ai_made_easy.core.vision import tasks  # noqa: E402, F401
from ai_made_easy.core.vision import template as _template  # noqa: E402

_template.register()
from ai_made_easy.core.vision import rules as _rules  # noqa: E402, F401
from ai_made_easy.core.vision import profile as _profile  # noqa: E402

_profile.register()
from ai_made_easy.core.vision import preview as _preview  # noqa: E402, F401
