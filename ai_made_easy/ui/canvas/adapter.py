"""CanvasController: the slim OdenGraphQt <-> core-IR adapter.

Responsibilities (and only these): node-class registration, canvas<->IR
conversion, view cosmetics (center/zoom/pipes/theme), validation coloring,
node status renames, composite splice, PNG render. Template filesystem IO
and demo data live elsewhere; the settle/debounce pipeline lives in
GraphService.
"""
from __future__ import annotations

from pathlib import Path

from OdenGraphQt import NodeGraph
from OdenGraphQt.constants import PipeLayoutEnum
from PySide6 import QtGui

from ai_made_easy.core.composites import Fragment
from ai_made_easy.core.graph import Edge, Graph, NodeInstance
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition
from ai_made_easy.ui.canvas import templates as template_store
from ai_made_easy.ui.canvas.node_factory import (
    BLOCK_TO_NODE_TYPE,
    NODE_TYPE_TO_BLOCK,
    make_node_class,
    node_type_for,
)

_ERROR_COLOR = (255, 73, 73, 255)


class CanvasController:
    def __init__(self) -> None:
        from ai_made_easy.ui.theme import CANVAS_BG, CANVAS_GRID

        self._graph = NodeGraph()
        self._graph.set_acyclic(True)
        self._graph.set_background_color(*CANVAS_BG)
        self._graph.set_grid_color(*CANVAS_GRID)
        self._graph.set_pipe_style(PipeLayoutEnum.CURVED.value)
        for block in get_registry().all():
            self._graph.register_node(make_node_class(block))
        template_store.register_user_templates(
            get_registry().register, make_node_class)
        # smart-mix wire guard: dtype-impossible wires are undone instantly
        self.guard_notifiers: list = []  # callables taking a friendly message
        self._graph.port_connected.connect(self._on_port_connected)

    # ------------------------------------------------------- wire guard

    def _port_dtype(self, port) -> str:  # noqa: ANN001
        node = port.node()
        defn = self.definition_of(node)
        if defn is None:
            return "tensor"
        ports = defn.inputs if port.type_() == "in" else defn.outputs
        spec = next((p for p in ports if p.name == port.name()), None)
        return spec.dtype if spec is not None else "tensor"

    def _on_port_connected(self, in_port, out_port) -> None:  # noqa: ANN001
        try:
            if self._port_dtype(in_port) == self._port_dtype(out_port):
                return
            names = (out_port.node().name(), in_port.node().name())
            out_port.disconnect_from(in_port)
            msg = (f"✋ undone: {names[0]} and {names[1]} speak different "
                   "languages (data vs settings) — they can't be wired together")
        except Exception:  # never break the canvas over a guard hiccup
            return
        for notify in self.guard_notifiers:
            notify(msg)

    # ------------------------------------------------------------ plumbing

    @property
    def widget(self):
        return self._graph.widget

    @property
    def node_graph(self) -> NodeGraph:
        return self._graph

    def definition_of(self, node) -> BlockDefinition | None:
        type_id = NODE_TYPE_TO_BLOCK.get(node.type_)
        return get_registry().get(type_id) if type_id else None

    def apply_theme(self, canvas_colors=None) -> None:
        if canvas_colors is None:
            from ai_made_easy.ui.theme import THEMES

            t = THEMES.get("dark")
            canvas_colors = (t["CANVAS_BG"], t["CANVAS_GRID"])
        bg, grid = canvas_colors
        self._graph.set_background_color(*bg)
        self._graph.set_grid_color(*grid)

    # ------------------------------------------------------ canvas <-> IR

    def to_ir(self, name: str = "untitled") -> Graph:
        reg = get_registry()
        g = Graph(name=name)
        for node in self._graph.all_nodes():
            type_id = NODE_TYPE_TO_BLOCK.get(node.type_)
            if type_id is None:
                continue  # backdrop / note nodes
            params = {}
            for spec in reg.get(type_id).params:
                params[spec.name] = node.get_property(spec.name)
            g.add_node(NodeInstance(node.id, type_id, params,
                                    (node.x_pos(), node.y_pos())))
        for node in self._graph.all_nodes():
            if node.type_ not in NODE_TYPE_TO_BLOCK:
                continue
            for out_port in node.output_ports():
                for connected in out_port.connected_ports():
                    g.add_edge(Edge(node.id, out_port.name(),
                                    connected.node().id, connected.name()))
        return g

    def clear(self) -> None:
        self._graph.clear_session()
        self._graph.clear_undo_stack()

    def load_ir(self, graph: Graph) -> None:
        self.clear()
        canvas_nodes = {}
        for inst in graph.nodes.values():
            cls_type = node_type_for(inst.type_id)
            if cls_type is None:
                raise ValueError(f"no canvas node registered for {inst.type_id}")
            node = self._graph.create_node(
                cls_type, name=inst.definition().display_name,
                pos=inst.position, selected=False)
            for key, value in inst.params.items():
                node.set_property(key, value)
            canvas_nodes[inst.instance_id] = node
        for edge in graph.edges:
            src = canvas_nodes[edge.source_id]
            dst = canvas_nodes[edge.target_id]
            port_out = src.outputs().get(edge.source_port)
            port_in = dst.inputs().get(edge.target_port)
            if port_out is not None and port_in is not None:
                port_out.connect_to(port_in)
        self.center_view()
        self.thicken_pipes()

    def load_demo(self, demo_path: Path | None = None) -> None:
        import json

        if demo_path is None:
            from ai_made_easy.ui.services.project_service import DEMO_SEED

            demo_path = DEMO_SEED
        self.load_ir(Graph.from_dict(json.loads(Path(demo_path).read_text())))

    def seed_demo(self) -> None:  # back-compat alias
        self.load_demo()

    # ------------------------------------------------------- view helpers

    def scene_center(self) -> tuple[float, float]:
        center = self._graph.viewer().viewport().rect().center()
        pos = self._graph.viewer().mapToScene(center)
        return pos.x(), pos.y()

    def center_view(self) -> None:
        """Fit the graph, but never below a scale where labels stay readable."""
        nodes = self._graph.all_nodes()
        if not nodes:
            return
        self._graph.set_zoom(-1)
        self._graph.center_on(nodes)
        viewer = self._graph.viewer()
        steps = 0
        while viewer.transform().m11() < 0.50 and steps < 12:
            self._graph.set_zoom(viewer.get_zoom() + 0.5)
            steps += 1

    def zoom(self, steps: int) -> None:
        self._graph.set_zoom(self._graph.viewer().get_zoom() + steps)

    def place_block(self, type_id: str) -> str:
        """Create a block at the viewport center; returns its display name."""
        cls_type = node_type_for(type_id)
        if cls_type is None:
            raise ValueError(f"unknown block type {type_id}")
        x, y = self.scene_center()
        node = self._graph.create_node(cls_type,
                                       name=get_registry().get(type_id).display_name,
                                       pos=(x, y))
        return node.name()

    def thicken_pipes(self) -> None:
        from OdenGraphQt.qgraphics.pipe import PipeItem

        for item in self._graph.viewer().scene().items():
            if isinstance(item, PipeItem):
                color = item.pen().color()
                item.set_pipe_styling((color.red(), color.green(), color.blue(),
                                       color.alpha()), width=2.6)

    # -------------------------------------------------- validation visuals

    def apply_validation(self, error_node_ids: set,
                         warning_node_ids: set | None = None,
                         shapes: dict | None = None,
                         issues: list | None = None) -> None:
        """Red ports + ✖ badge on error nodes, ⚠ badge on warnings; family
        shade otherwise. The tooltip carries the output shape (Orange's
        state-summary idea) plus this node's issues (its severity-icon
        tooltip idea)."""
        warning_node_ids = warning_node_ids or set()
        msgs: dict[str, list[str]] = {}
        for i in issues or []:
            if i.node_id:
                glyph = "✖" if i.severity == "error" else "⚠"
                msgs.setdefault(i.node_id, []).append(f"{glyph} {i.message}")
        for node in self._graph.all_nodes():
            node_def = self.definition_of(node)
            badge = None
            if node.id in error_node_ids:
                color = _ERROR_COLOR
                badge = "error"
            elif node.id in warning_node_ids:
                color = (QtGui.QColor(node_def.color).darker(135).getRgb()
                         if node_def is not None else (180, 180, 90, 255))
                badge = "warning"
            elif node_def is not None:
                color = QtGui.QColor(node_def.color).darker(135).getRgb()
            else:
                color = (180, 180, 90, 255)
            for port in [*node.input_ports(), *node.output_ports()]:
                port.color = color
            view = node.view
            view._aime_badge = badge
            lines = [node_def.display_name if node_def else node.name()]
            if shapes and node.id in shapes:
                lines.append(f"output {list(shapes[node.id])}")
            lines += msgs.get(node.id, [])
            view.setToolTip("\n".join(lines))
            view.update()

    def nodes_of_type(self, block_type_id: str) -> list:
        node_type = node_type_for(block_type_id)
        if node_type is None:
            return []
        return [n for n in self._graph.all_nodes() if n.type_ == node_type]

    def set_node_status(self, block_type_id: str, text: str | None,
                        default_name: str) -> None:
        """Show run status in the node's own title (None restores)."""
        for node in self.nodes_of_type(block_type_id):
            node.set_name(text if text else default_name)

    def set_node_progress(self, block_type_id: str,
                          fraction: float | None) -> None:
        """Bottom progress strip on a block (None clears)."""
        for node in self.nodes_of_type(block_type_id):
            node.view._aime_progress = fraction
            node.view.update()

    def set_wire_flow(self, running: bool) -> None:
        """Marching-ants dashes along every wire while a run is active."""
        from ai_made_easy.ui.canvas import painter

        if getattr(self, "_flow_animator", None) is None:
            self._flow_animator = painter.WireFlowAnimator()
        self._flow_animator.set_running(running)

    def select_and_center(self, node_id: str) -> None:
        """Select a block by IR id and bring it into view (issue jump-to)."""
        for node in self._graph.all_nodes():
            if node.id == node_id:
                node.setSelected(True)
                self._graph.center_on([node])
                return

    # -------------------------------------------------------- composites

    def expand_selected(self) -> int:
        count = 0
        for node in list(self._graph.selected_nodes()):
            defn = self.definition_of(node)
            if defn is None or defn.builder is None:
                continue
            params = {}
            for spec in defn.params:
                params[spec.name] = node.get_property(spec.name)
            try:
                frag = defn.builder(defn.default_params() | params)
            except Exception as exc:
                raise ValueError(f"cannot expand {defn.display_name}: {exc}") from exc
            self._splice_fragment(node, frag)
            count += 1
        return count

    def _node_params(self, node) -> dict:
        defn = self.definition_of(node)
        if defn is None:
            return {}
        return {spec.name: node.get_property(spec.name) for spec in defn.params}

    def _splice_fragment(self, node, frag: Fragment) -> None:
        ext_in = [cp for port in node.input_ports() for cp in port.connected_ports()]
        ext_out = [cp for port in node.output_ports() for cp in port.connected_ports()]

        first = frag.nodes[0]["position"]
        min_y = min(nd["position"][1] for nd in frag.nodes)
        ox = node.x_pos() - first[0]
        oy = node.y_pos() - min_y + (first[1] - min_y)

        prefix = f"n{len(self._graph.all_nodes())}_"
        id_map: dict[str, object] = {}
        for nd in frag.nodes:
            defn = get_registry().get(nd["type"])
            new_node = self._graph.create_node(
                node_type_for(nd["type"]), name=defn.display_name,
                pos=(nd["position"][0] + ox, nd["position"][1] + oy),
                selected=False)
            for key, value in nd.get("params", {}).items():
                new_node.set_property(key, value)
            id_map[nd["id"]] = new_node

        for src, src_port, dst, dst_port in frag.edges:
            out_p = id_map[src].outputs().get(src_port)
            in_p = id_map[dst].inputs().get(dst_port)
            if out_p is not None and in_p is not None:
                out_p.connect_to(in_p)

        entry = id_map[frag.entry]
        exit_ = id_map[frag.exit]
        for cp in ext_in:
            cp.connect_to(entry.input_ports()[0])
        for tp in ext_out:
            exit_.output_ports()[0].connect_to(tp)

        self._graph.delete_node(node)
        self.thicken_pipes()

    # --------------------------------------------------- selection -> IR

    def selection_fragment(self) -> tuple[Fragment, Graph]:
        """Selected nodes + internal edges as a Fragment (validates boundaries).

        Returns (fragment, ir) — ir is the full snapshot for edge scanning.
        """
        sel = self._graph.selected_nodes()
        if len(sel) < 2:
            raise ValueError("select at least two connected blocks to template")
        sel_ids = {n.id for n in sel}
        nodes, edges = [], []
        for n in sel:
            defn = self.definition_of(n)
            if defn is None or defn.builder is not None:
                raise ValueError("templates cannot contain architecture macros")
            nodes.append({"id": n.id, "type": defn.type_id,
                          "params": self._node_params(n),
                          "position": (0.0, 0.0)})
        ir = self.to_ir("selection")
        for e in ir.edges:
            if e.source_id in sel_ids and e.target_id in sel_ids:
                edges.append((e.source_id, e.source_port,
                              e.target_id, e.target_port))
        entry_nodes = [n.id for n in sel
                       if any(e.target_id == n.id and e.source_id not in sel_ids
                              for e in ir.edges)]
        exit_nodes = [n.id for n in sel
                      if any(e.source_id == n.id and e.target_id not in sel_ids
                             for e in ir.edges)]
        if len(entry_nodes) != 1 or len(exit_nodes) != 1:
            raise ValueError(
                "a template needs exactly one input boundary and one output "
                f"boundary (found {len(entry_nodes)} in / {len(exit_nodes)} out)")
        return Fragment(nodes=nodes, edges=edges,
                        entry=entry_nodes[0], exit=exit_nodes[0]), ir

    def save_selection_as_template(self, name: str) -> Path:
        frag, _ir = self.selection_fragment()
        path = template_store.save_template(frag, name)
        template_store.register_user_templates(
            get_registry().register, make_node_class)
        return path

    # --------------------------------------------------------- png export

    def export_canvas_png(self, path, on_done=None) -> None:
        """Deferred whole-scene render (needs the event loop)."""
        from PySide6 import QtCore, QtGui

        viewer = self._graph.viewer()
        rect = viewer.scene().itemsBoundingRect().adjusted(-40, -40, 40, 40)
        if rect.width() <= 0 or rect.height() <= 0:
            if on_done:
                on_done(None, "canvas is empty — nothing to export")
            return
        old_transform = viewer.transform()
        viewer.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        viewer.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        vp = viewer.viewport().size()
        scale = min(vp.width() / rect.width(), vp.height() / rect.height())
        fit = QtGui.QTransform()
        fit.scale(scale, scale).translate(-rect.left(), -rect.top())
        viewer.setTransform(fit)

        def grab_and_restore() -> None:
            pixmap = viewer.grab()
            viewer.setTransform(old_transform)
            error = None if pixmap.save(str(path)) else f"could not write {path}"
            if on_done:
                on_done(path if not error else None, error)

        QtCore.QTimer.singleShot(150, grab_and_restore)
