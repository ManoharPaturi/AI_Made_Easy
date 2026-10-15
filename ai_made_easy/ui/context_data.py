"""Data: open, profile, split and augment.

The ``act_*`` slots of :class:`~ai_made_easy.ui.context.AppContext` for this area,
as a mixin: the methods use the stores, services and panels AppContext builds (split out of
context.py in 3.3; no behaviour change).
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets

from ai_made_easy.core.graph import Graph


class DataActions:
    # ================================================================ data

    def act_data(self, *_):
        if self.window is not None:
            self.window.docks["data"].show()
            self.window.docks["data"].raise_()
        self._profile_selected()

    def _refresh_datasets(self, ir: Graph) -> None:
        from ai_made_easy.core.data.lints import dataset_nodes

        items = []
        for node_id, type_id, params in dataset_nodes(ir):
            source = params.get("root") or params.get("path") or params.get("dataset") or \
                params.get("repo_id") or params.get("kind") or ""
            items.append((node_id, f"{self.canvas.node_title(node_id)} · {source}"))
        previous = self.data_page.current_dataset()
        self.data_page.set_datasets(items)
        if not items:
            self._data_key = None
            self.data_page.clear()
        elif self.data_page.current_dataset() == previous:
            self._profile_selected()  # params may have changed

    def _profile_selected(self, refresh: bool = False) -> None:
        node_id = self.data_page.current_dataset()
        ir = self._last_ir
        if not node_id or ir is None or node_id not in ir.nodes:
            return
        node = ir.nodes[node_id]
        key = self.data_service.cache.key(node.type_id, dict(node.resolved_params()),
                                          self.data_service.base)
        if key == self._data_key and not refresh and self.data_page.profile is not None:
            return
        self._data_key = key
        self.data_page.set_loading()
        self.data_service.request(node.type_id, dict(node.resolved_params()), refresh=refresh)

    def _on_profiled(self, key, profile) -> None:  # noqa: ANN001
        if key == self._data_key:
            self.data_page.set_profile(profile)

    def _load_rows(self) -> None:
        if self.data_page.profile is not None:
            self.data_service.load_rows(self._data_key, self.data_page.profile)

    def act_profile_file(self, *_):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Profile a Data File", str(self._last_dir()),
            "Data files (*.csv *.tsv *.parquet *.xlsx *.xls *.json *.jsonl *.npz);;"
            "All files (*)")
        if not path:
            return
        self.data_page.set_loading(f"Profiling {Path(path).name}…")
        self._data_key = self.data_service.request_path(path)

    def act_data_split(self, *_):
        self.act_data()
        self.data_page.tabs.setCurrentIndex(4)
        self.act_split_preview()

    def act_data_augment(self, *_):
        self.act_data()
        self.data_page.tabs.setCurrentIndex(5)
        self.act_augmentation_preview()

    def act_split_preview(self, *_):
        if self._last_ir is None:
            return
        self.data_page.split_btn.setEnabled(False)
        self.data_page.split_info.setText("Computing the split…")
        self.data_service.split(self._last_ir)

    def act_augmentation_preview(self, *_):
        if self._last_ir is None:
            return
        self.data_page.augment_btn.setEnabled(False)
        self.data_page.augment_info.setText("Rendering training views…")
        self.data_service.augment(self._last_ir)
