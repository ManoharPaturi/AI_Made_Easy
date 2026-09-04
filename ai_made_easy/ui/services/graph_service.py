"""GraphService: the IR bridge + the ONE settle pipeline.

Owns the debounce timer; every canvas mutation funnels through
graph_settled(Graph) exactly once per change. Also: validation, block
placement, composite expansion, node status renames.
"""
from __future__ import annotations

from PySide6 import QtCore

from ai_made_easy.core.graph import Graph
from ai_made_easy.ui.canvas import CanvasController


class GraphService(QtCore.QObject):
    graph_settled = QtCore.Signal(object)  # Graph (core IR)
    guard_message = QtCore.Signal(str)     # wire-guard friendly notices

    def __init__(self, adapter: CanvasController, log, parent=None):
        super().__init__(parent)
        self.adapter = adapter
        self.log = log
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.settle_now)
        self._loading = False  # load-guard: suppress settle during load
        self._node_defaults: dict[str, str] = {}  # type_id -> original name
        self.last_shapes: dict = {}

        adapter.guard_notifiers.append(self.guard_message.emit)

        graph = adapter.node_graph
        graph.node_created.connect(lambda *_: self.schedule())
        graph.nodes_deleted.connect(lambda *_: self.schedule())
        graph.port_connected.connect(lambda *_: self.schedule())
        graph.port_disconnected.connect(lambda *_: self.schedule())
        graph.property_changed.connect(lambda *_: self.schedule())

    # --------------------------------------------------------- pipeline

    def schedule(self) -> None:
        if not self._loading:
            self._timer.start()

    def settle_now(self) -> None:
        self._timer.stop()
        try:
            ir = self.snapshot()
        except Exception as exc:  # mid-edit states; consumers show & move on
            self.log.error(f"validation error: {exc}")
            return
        self.graph_settled.emit(ir)

    def snapshot(self) -> Graph:
        from ai_made_easy.ui.stores import ProjectStore

        return self.adapter.to_ir()

    def load(self, graph: Graph) -> None:
        """Replace canvas content; settle fires once, without dirty marking."""
        self._loading = True
        try:
            self.adapter.load_ir(graph)
        finally:
            self._loading = False
        self.settle_now()

    # ------------------------------------------------------- validation

    @staticmethod
    def errors_of(graph: Graph) -> list:
        return [i for i in graph.validate() if i.severity == "error"]

    @staticmethod
    def validate_dict(data: dict) -> tuple[bool, str]:
        """Shared severity filter (assistant + project service use this)."""
        try:
            graph = Graph.from_dict(data)
        except Exception as exc:
            return False, f"invalid graph JSON: {exc}"
        errors = GraphService.errors_of(graph)
        if errors:
            return False, "\n".join(str(i) for i in errors)
        return True, ""

    def apply_validation(self, issues: list) -> None:
        error_ids = {i.node_id for i in issues if i.severity == "error" and i.node_id}
        warn_ids = {i.node_id for i in issues
                    if i.severity == "warning" and i.node_id}
        self.adapter.apply_validation(error_ids, warn_ids,
                                      self.last_shapes, issues)

    def note_shapes(self, ir: Graph) -> None:
        """Cache the latest per-node output shapes for tooltips."""
        try:
            self.last_shapes, _issues = ir.infer_shapes_detailed()
        except Exception:
            self.last_shapes = {}

    # ---------------------------------------------------- canvas actions

    def place_block(self, type_id: str) -> None:
        try:
            name = self.adapter.place_block(type_id)
            self.log.info(f"placed {name}")
        except Exception as exc:
            self.log.error(str(exc))

    def expand_selected(self) -> None:
        try:
            count = self.adapter.expand_selected()
        except Exception as exc:
            self.log.error(str(exc))
            return
        if count:
            self.log.info(f"expanded {count} architecture block(s) into primitives")
            self.settle_now()
        else:
            self.log.info("no architecture blocks in the selection "
                          "(drag one from the Architectures/Custom palette first)")

    def save_selection_template(self, name: str):
        return self.adapter.save_selection_as_template(name)

    # ------------------------------------------------------- run status

    def set_node_status(self, type_id: str, text: str | None) -> None:
        """Show run status on a block's title; None restores the default."""
        nodes = self.adapter.nodes_of_type(type_id)
        if text is None:
            default = self._node_defaults.pop(type_id, None)
            if default is None:
                return
            self.adapter.set_node_status(type_id, None, default)
            return
        if type_id not in self._node_defaults and nodes:
            self._node_defaults[type_id] = nodes[0].name()
        self.adapter.set_node_status(type_id, text,
                                     self._node_defaults.get(type_id, type_id))
