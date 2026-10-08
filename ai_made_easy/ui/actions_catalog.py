"""Actions as data: every menu command, shortcut and toolbar entry declared once.

The Workbench builds QActions from these specs, menus assemble them, the
command palette searches them and the shortcuts sheet documents them. The
completeness test asserts every slot resolves on the AppContext.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    id: str                 # stable objectName
    text: str
    menu: str               # "" = not in a menu
    slot: str               # method name on AppContext
    shortcut: str = ""
    tooltip: str = ""
    checkable: bool = False
    separator_after: bool = False
    icon: str = ""


CATALOG: list[ActionSpec] = [
    # ---- File ----
    ActionSpec("file.new", "&New Project", "&File", "act_new", "Ctrl+N", icon="new"),
    ActionSpec("file.open", "&Open Project…", "&File", "act_open", "Ctrl+O", icon="open"),
    ActionSpec("file.examples", "Open &Example…", "&File", "act_samples", "Ctrl+Shift+O",
               "Open a ready-made example project", separator_after=True),
    ActionSpec("file.import_model", "&Import Model…", "&File", "act_import_model",
               "Ctrl+Shift+I", "Turn a PyTorch, ONNX or Keras model into editable blocks",
               icon="download", separator_after=True),
    ActionSpec("file.save", "&Save", "&File", "act_save", "Ctrl+S", icon="save"),
    ActionSpec("file.save_as", "Save &As…", "&File", "act_save_as", "Ctrl+Shift+S",
               separator_after=True),
    ActionSpec("file.export_archive", "Export Project &Archive…", "&File",
               "act_export_bundle", "",
               "One .aime file with the graph, model card and last run artefacts"),
    ActionSpec("file.open_archive", "Open Project A&rchive…", "&File", "act_open_bundle",
               separator_after=True),
    ActionSpec("file.export_png", "Export Canvas as &Image…", "&File", "act_export_png",
               icon="image", separator_after=True),
    ActionSpec("file.quit", "&Quit", "&File", "act_quit", "Ctrl+Q"),

    # ---- Edit ----
    ActionSpec("edit.undo", "&Undo", "&Edit", "act_undo", "Ctrl+Z", icon="undo"),
    ActionSpec("edit.redo", "&Redo", "&Edit", "act_redo", "Ctrl+Shift+Z", icon="redo",
               separator_after=True),
    ActionSpec("edit.delete", "&Delete Selection", "&Edit", "act_delete", "Delete",
               icon="trash"),
    ActionSpec("edit.select_all", "Select &All", "&Edit", "act_select_all", "Ctrl+A",
               separator_after=True),
    ActionSpec("edit.find", "&Find Block…", "&Edit", "act_find", "Ctrl+F", icon="search"),

    # ---- View ----
    ActionSpec("view.theme_dark", "&Dark Theme", "&View", "act_theme_dark", checkable=True,
               icon="moon"),
    ActionSpec("view.theme_light", "&Light Theme", "&View", "act_theme_light",
               checkable=True, icon="sun", separator_after=True),
    ActionSpec("view.zoom_in", "Zoom &In", "&View", "act_zoom_in", "Ctrl+=", icon="zoom_in"),
    ActionSpec("view.zoom_out", "Zoom &Out", "&View", "act_zoom_out", "Ctrl+-",
               icon="zoom_out"),
    ActionSpec("view.zoom_fit", "&Fit to View", "&View", "act_zoom_fit", "Ctrl+0", icon="fit"),
    ActionSpec("view.auto_layout", "Auto-&Arrange Blocks", "&View", "act_auto_layout",
               "Ctrl+L", "Lay blocks out left to right by data flow", separator_after=True),

    # ---- Model ----
    ActionSpec("model.validate", "&Validate", "&Model", "act_validate", "Ctrl+Shift+V",
               "Run all checks now", icon="check"),
    ActionSpec("model.expand", "&Expand Architecture", "&Model", "act_expand", "Ctrl+E",
               "Replace the selected architecture block with its layers", icon="expand"),
    ActionSpec("model.save_selection", "Save Selection as &Block…", "&Model",
               "act_save_selection", "Ctrl+Shift+B",
               "Turn the selected blocks into a reusable custom block", icon="plus_box"),
    ActionSpec("model.export_blocks", "Export Custom Blocks…", "&Model", "act_export_blocks",
               tooltip="Share your custom blocks as a .aimeblocks pack"),
    ActionSpec("model.import_blocks", "Import Custom Blocks…", "&Model", "act_import_blocks",
               tooltip="Install custom blocks from a .aimeblocks pack"),

    # ---- Run ----
    ActionSpec("run.train", "&Train", "&Run", "act_train", "Ctrl+R",
               "Generate the training script and run it", icon="play"),
    ActionSpec("run.test", "Test &Forward Pass", "&Run", "act_test_run", "Ctrl+T",
               "Build the model and run one batch through it", icon="zap"),
    ActionSpec("run.stop", "&Stop", "&Run", "act_stop", "Ctrl+.", icon="stop",
               separator_after=True),
    ActionSpec("run.sweep", "New Hyperparameter S&weep…", "&Run", "act_new_sweep",
               "Ctrl+Shift+R", "Search parameter values over several training runs",
               icon="sparkle"),
    ActionSpec("run.experiments", "E&xperiments", "&Run", "act_experiments", "Ctrl+Shift+X",
               "Run history, comparison and sweeps", icon="chart", separator_after=True),
    ActionSpec("run.errors", "&Error Analysis", "&Run", "act_error_analysis",
               icon="list"),
    ActionSpec("run.saliency", "&Saliency Maps", "&Run", "act_saliency", icon="eye"),
    ActionSpec("run.card", "&Model Card", "&Run", "act_model_card", icon="card"),
    ActionSpec("run.folder", "Open Run &Folder", "&Run", "act_open_run_folder", icon="folder"),

    # ---- Export ----
    ActionSpec("export.pytorch_model", "PyTorch &Model (.py)", "E&xport", "act_export_pytorch_model",
               "Ctrl+Shift+E", icon="code"),
    ActionSpec("export.pytorch_train", "PyTorch &Training Script (.py)", "E&xport",
               "act_export_pytorch_train", separator_after=True),
    ActionSpec("export.keras_model", "&Keras Model (.py)", "E&xport", "act_export_keras_model"),
    ActionSpec("export.keras_train", "Keras Training Script (.py)", "E&xport",
               "act_export_keras_train", separator_after=True),
    ActionSpec("export.sklearn", "&scikit-learn Pipeline (.py)", "E&xport",
               "act_export_sklearn", separator_after=True),
    ActionSpec("export.onnx", "&ONNX Architecture (.onnx)", "E&xport", "act_export_onnx",
               tooltip="The designed network with initial weights; trained models export "
                       "through Model Server"),
    ActionSpec("export.jit", "Torch&Script Architecture (.pt)", "E&xport", "act_export_jit",
               tooltip="The designed network with initial weights; trained models export "
                       "through Model Server"),
    ActionSpec("export.web", "&Web Demo (.html)", "E&xport", "act_export_web",
               "Single-file browser demo with the trained weights", separator_after=True),
    ActionSpec("export.deploy", "Model &Server (FastAPI + Docker)…", "E&xport", "act_deploy",
               "Ctrl+Shift+D", "Package the trained model as a deployable web service",
               icon="package", separator_after=True),
    ActionSpec("export.llm", "&LLM Workflow Script (.py)", "E&xport", "act_export_llm"),

    # ---- Help ----
    ActionSpec("help.shortcuts", "&Keyboard Shortcuts", "&Help", "act_shortcuts",
               icon="keyboard"),
    ActionSpec("help.about", "&About AI Made Easy", "&Help", "act_about", icon="help"),
]

MENU_ORDER = ["&File", "&Edit", "&View", "&Model", "&Run", "E&xport", "&Help"]

TOOLBAR = ["file.new", "file.open", "file.save", "|", "edit.undo", "edit.redo", "|",
           "model.validate", "run.test", "run.train", "run.stop"]


def build_actions(context, parent) -> dict[str, object]:
    """Create QActions from the catalog; objectName = spec.id."""
    from PySide6 import QtGui

    from ai_made_easy.ui import icons

    actions: dict = {}
    for spec in CATALOG:
        action = QtGui.QAction(spec.text, parent)
        action.setObjectName(spec.id)
        if spec.shortcut:
            seq = QtGui.QKeySequence(spec.shortcut)
            if spec.id == "edit.delete":
                action.setShortcuts([seq, QtGui.QKeySequence("Backspace")])
            else:
                action.setShortcut(seq)
        if spec.tooltip:
            action.setToolTip(spec.tooltip)
            action.setStatusTip(spec.tooltip)
        if spec.icon:
            action.setIcon(icons.icon(spec.icon))
        action.setCheckable(spec.checkable)
        handler = getattr(context, spec.slot, None)
        if handler is not None:
            action.triggered.connect(handler)
        actions[spec.id] = action
    return actions
