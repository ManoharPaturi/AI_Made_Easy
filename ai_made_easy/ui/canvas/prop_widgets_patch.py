"""Properties-panel patches — bounds that actually reach the widgets.

The stock ``PropSpinBox``/``PropDoubleSpinBox`` have no ``set_min``/``set_max``
(the bin calls them whenever a property declares a range → AttributeError,
so the panel never populates for blocks with int params), and the double
spinbox keeps Qt's 0.00–99.99 / 2-decimal defaults, silently rounding
``lr=0.001`` to ``0.0``. We patch both, in the canvas boundary like painter.py.
"""
from __future__ import annotations

import math

from OdenGraphQt.custom_widgets.properties_bin.prop_widgets_base import (
    PropDoubleSpinBox,
    PropSpinBox,
)


def install_prop_widget_patches() -> None:
    if getattr(PropSpinBox, "_aime_patched", False):
        return
    PropSpinBox._aime_patched = True

    PropSpinBox.set_min = lambda self, v: self.setMinimum(int(v))
    PropSpinBox.set_max = lambda self, v: self.setMaximum(int(v))

    _orig_dsb_init = PropDoubleSpinBox.__init__

    def _dsb_init(self, parent=None) -> None:  # noqa: ANN001
        _orig_dsb_init(self, parent)
        self.setDecimals(6)
        self.setRange(-1_000_000_000.0, 1_000_000_000.0)

    PropDoubleSpinBox.__init__ = _dsb_init
    PropDoubleSpinBox.set_min = lambda self, v: self.setMinimum(float(v))
    PropDoubleSpinBox.set_max = lambda self, v: self.setMaximum(float(v))

    _orig_set_value = PropDoubleSpinBox.set_value

    def _dsb_set_value(self, value) -> None:  # noqa: ANN001
        # grow decimals so tiny values (eps 1e-8, lr 1e-4) survive the spinbox
        if isinstance(value, (int, float)) and value not in (0, 0.0):
            try:
                need = max(2, min(10, -math.floor(math.log10(abs(value))) + 2))
            except ValueError:
                need = 2
            if need > self.decimals():
                self.setDecimals(need)
        _orig_set_value(self, value)

    PropDoubleSpinBox.set_value = _dsb_set_value
