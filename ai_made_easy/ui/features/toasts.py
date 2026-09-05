"""ToastLayer: floating top-centre notifications (the app's applause).

One transparent overlay child of the main window; every ``status_message``
also lands here as a soft pill that fades in, lives ~3s, fades out, and
dismisses on click. The statusbar keeps the persistent record — toasts
are the moment of feedback.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

_LIFE_MS = 3000        # visible lifetime before fading
_FADE_MS = 350
_MAX_TOASTS = 3        # older toasts are retired first


class Toast(QtWidgets.QLabel):
    """A single notification pill."""

    closed = QtCore.Signal()

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("toast")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._effect = QtWidgets.QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)
        self._fade = QtCore.QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.finished.connect(self._on_faded)

    def enter(self) -> None:
        self._fade.stop()
        self._fade.setDuration(_FADE_MS)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

    def leave(self) -> None:
        self._fade.stop()
        self._fade.setDuration(_FADE_MS)
        self._fade.setStartValue(self._effect.opacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def mousePressEvent(self, _event) -> None:  # noqa: N802 (Qt name)
        self.leave()

    def _on_faded(self) -> None:
        if self._effect.opacity() == 0.0:
            self.closed.emit()
            self.deleteLater()


class ToastLayer(QtWidgets.QWidget):
    """Overlay that stacks toasts under the header, top-centre."""

    def __init__(self, host: QtWidgets.QWidget, parent=None):
        super().__init__(host if parent is None else parent)
        self._host = host
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self._column = QtWidgets.QVBoxLayout(self)
        self._column.setContentsMargins(0, 12, 0, 0)
        self._column.setSpacing(8)
        self._column.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self._column.addStretch(1)
        self.raise_()
        host.installEventFilter(self)
        self._sync_geometry()

    # ------------------------------------------------------------- api
    def toast(self, text: str) -> None:
        # retire oldest before adding, so at most _MAX_TOASTS are visible
        while self._column.count() - 1 >= _MAX_TOASTS:
            item = self._column.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
                item.widget().deleteLater()
        pill = Toast(text, self)
        pill.closed.connect(lambda: self._column.removeWidget(pill))
        self._column.insertWidget(self._column.count() - 1, pill,
                                  alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        pill.enter()
        QtCore.QTimer.singleShot(_LIFE_MS, pill.leave)

    # --------------------------------------------------------- plumbing
    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt name)
        if watched is self._host and event.type() in (
                QtCore.QEvent.Type.Resize,
                QtCore.QEvent.Type.ChildAdded):
            self._sync_geometry()
            self.raise_()
        return super().eventFilter(watched, event)

    def _sync_geometry(self) -> None:
        # sit just below the (unified) header, spanning the window
        top = 86 if self._host.property("unifiedHeader") else 64
        self.setGeometry(0, top, self._host.width(),
                         max(0, self._host.height() - top))
