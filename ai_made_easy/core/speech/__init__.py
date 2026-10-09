"""Speech: keyword spotting, CTC speech recognition and audio tagging.

Importing the package registers the blocks, the tasks and their resolver, the
training loop, the design rules and the data profiler.
"""
from ai_made_easy.core.speech import blocks as _blocks

_blocks.register_all()

from ai_made_easy.core.speech import tasks  # noqa: E402, F401
from ai_made_easy.core.speech import template as _template  # noqa: E402

_template.register()
from ai_made_easy.core.speech import rules as _rules  # noqa: E402, F401
from ai_made_easy.core.speech import profile as _profile  # noqa: E402

_profile.register()
