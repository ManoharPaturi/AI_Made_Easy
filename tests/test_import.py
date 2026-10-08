"""Model import: PyTorch (torch.fx), ONNX and Keras round trips with numeric checks."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))

torch = pytest.importorskip("torch")

from ai_made_easy.core.codegen import CodegenError, generate  # noqa: E402
from ai_made_easy.core.graph import Graph  # noqa: E402
from ai_made_easy.core.importers import ModelImportError, import_model  # noqa: E402


def _check(result: dict, exact: bool = True) -> Graph:
    assert not result["unsupported"], result["unsupported"]
    v = result["verification"]
    assert v["ok"], v
    if exact:
        assert v["outputs_match"], v
        assert v["weights_missing"] == 0
    graph = Graph.from_dict(result["graph"])
    assert [i for i in graph.validate() if i.severity == "error"] == []
    compile(generate(graph, "pytorch"), "<gen>", "exec")
    try:  # Keras export works unless a block has no Keras form for these shapes
        compile(generate(graph, "keras"), "<gen>", "exec")
    except CodegenError as exc:
        assert "Keras" in str(exc)
    return graph


# ------------------------------------------------------------------ PyTorch

@pytest.mark.parametrize("factory,shape,dtype", [
    ("TextLSTM", [20], "int64"),
    ("TinyTransformer", [10, 16], "float32"),
    ("SkipUNet", [1, 32, 32], "float32"),
    ("FunctionalMLP", [20], "float32"),
])
def test_torch_modules_round_trip_exactly(factory, shape, dtype):
    import import_models

    from ai_made_easy.core.importers.torch_fx import import_module

    model = getattr(import_models, factory)()
    result = import_module(model, shape, dtype)
    _check(result)
    assert result["verification"]["parameters"] == result["original"]["parameters"]


@pytest.mark.parametrize("arch", ["resnet18", "mobilenet_v2", "vgg11_bn", "densenet121"])
def test_torchvision_architectures(arch):
    tv = pytest.importorskip("torchvision")
    from ai_made_easy.core.importers.torch_fx import import_module

    model = getattr(tv.models, arch)()
    result = import_module(model, [3, 64, 64])
    if arch == "densenet121":  # functional relu(inplace) + cat chains: still exact
        assert not result["unsupported"], result["unsupported"][:3]
    _check(result)


def test_unsupported_operations_are_reported_not_guessed():
    import import_models

    from ai_made_easy.core.importers.torch_fx import Unsupported, import_module

    result = import_module(import_models.Bilinear(), [8])
    assert any("matmul" in u for u in result["unsupported"]), result["unsupported"]
    assert "verification" not in result
    with pytest.raises(Unsupported, match="control flow"):
        import_module(import_models.Branchy(), [4])


def test_subprocess_import_from_a_python_file(tmp_path):
    weights = tmp_path / "w.pt"
    result = import_model("pytorch", source=str(FIXTURES / "import_models.py"),
                          attr="FunctionalMLP", kwargs={"hidden": 16}, input_shape=[20],
                          weights_out=str(weights), name="mlp")
    assert result.ok and result.graph["name"] == "mlp"
    assert result.verification["outputs_match"]
    assert "identical" in result.summary()
    # the exported weights load into the generated model file
    graph = Graph.from_dict(result.graph)
    namespace: dict = {"__name__": "x"}
    exec(compile(generate(graph, "pytorch"), "<gen>", "exec"), namespace)  # noqa: S102
    namespace["build_model"]().load_state_dict(torch.load(weights))
    with pytest.raises(ModelImportError, match="input shape"):
        import_model("pytorch", source="x.py", attr="Net")
    with pytest.raises(ModelImportError):
        import_model("pytorch", source=str(FIXTURES / "import_models.py"), attr="Nope",
                     input_shape=[4])


# --------------------------------------------------------------------- ONNX

@pytest.mark.parametrize("name", ["SkipUNet", "FunctionalMLP", "resnet18"])
def test_onnx_round_trip(name, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import import_models

    from ai_made_easy.core.importers.onnx_import import import_onnx

    if name == "resnet18":
        model, shape = pytest.importorskip("torchvision").models.resnet18(), (3, 64, 64)
    else:
        model = getattr(import_models, name)()
        shape = (1, 32, 32) if name == "SkipUNet" else (20,)
    path = tmp_path / f"{name}.onnx"
    torch.onnx.export(model.eval(), (torch.randn(1, *shape),), str(path), input_names=["x"],
                      output_names=["y"], dynamic_axes={"x": {0: "b"}, "y": {0: "b"}},
                      dynamo=False, opset_version=17)
    _check(import_onnx(str(path)))


def test_onnx_fixed_batch_size(tmp_path):
    """Exports without dynamic axes accept only their batch size (usually 1)."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import import_models

    from ai_made_easy.core.importers.onnx_import import import_onnx

    path = tmp_path / "fixed.onnx"
    torch.onnx.export(import_models.FunctionalMLP().eval(), (torch.randn(1, 20),), str(path),
                      input_names=["x"], dynamo=False, opset_version=17)
    _check(import_onnx(str(path)))


# -------------------------------------------------------------------- Keras

def _keras_models():
    keras = pytest.importorskip("keras")
    L = keras.layers

    def cnn():
        return keras.Sequential([
            keras.Input((28, 28, 1)), L.Conv2D(8, 3, padding="same", activation="relu"),
            L.BatchNormalization(), L.MaxPooling2D(), L.Conv2D(16, 3, activation="relu"),
            L.Flatten(), L.Dropout(0.2), L.Dense(10, activation="softmax")], name="cnn")

    def text():
        return keras.Sequential([
            keras.Input((20,), dtype="int32"), L.Embedding(500, 16),
            L.LSTM(32, return_sequences=True), L.GRU(16), L.Dense(3)], name="text")

    def bidirectional():
        return keras.Sequential([
            keras.Input((10, 6)), L.Bidirectional(L.LSTM(8, return_sequences=True)),
            L.LayerNormalization(), L.SimpleRNN(4)], name="bi")

    def residual():
        x = keras.Input((16, 16, 4))
        y = L.ReLU()(L.Conv2D(4, 3, padding="same")(x))
        merged = L.Concatenate()([L.Add()([x, y]), x])
        return keras.Model(x, L.Dense(2)(L.GlobalMaxPooling2D()(merged)), name="res")

    def conv1d():
        return keras.Sequential([
            keras.Input((32, 3)), L.Conv1D(8, 5, padding="same", activation="relu"),
            L.MaxPooling1D(2), L.LSTM(8), L.Dense(1)], name="c1d")

    return [cnn, text, bidirectional, residual, conv1d]


@pytest.mark.parametrize("index", range(5))
def test_keras_round_trip(index, tmp_path):
    os.environ.setdefault("KERAS_BACKEND", "torch")
    builder = _keras_models()[index]
    from ai_made_easy.core.importers.keras_import import import_keras

    model = builder()
    path = tmp_path / f"{model.name}.keras"
    model.save(path)
    _check(import_keras(str(path)))


def test_keras_unsupported_layer_is_reported(tmp_path):
    keras = pytest.importorskip("keras")
    from ai_made_easy.core.importers.keras_import import import_keras

    model = keras.Sequential([keras.Input((8,)), keras.layers.Dense(4),
                              keras.layers.UnitNormalization()])
    path = tmp_path / "m.keras"
    model.save(path)
    result = import_keras(str(path))
    assert any("UnitNormalization" in u for u in result["unsupported"])


# ---------------------------------------------------------------- CLI / UI

def test_cli_import(tmp_path):
    out = tmp_path / "project.json"
    proc = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "import", "pytorch",
                           str(FIXTURES / "import_models.py"), "--attr", "SkipUNet",
                           "--shape", "1,32,32", "-o", str(out)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "identical" in proc.stdout
    graph = Graph.from_dict(json.loads(out.read_text()))
    assert any(n.type_id == "core.concatenate" for n in graph.nodes.values())


def test_import_dialog_and_open(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from ai_made_easy.ui.features.import_model import ImportModelDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = ImportModelDialog(None)
    with pytest.raises(ValueError, match="model file"):
        dialog.request()
    dialog.source.setText(str(FIXTURES / "import_models.py"))
    with pytest.raises(ValueError, match="class"):
        dialog.request()
    dialog.attr.setText("SkipUNet")
    dialog.shape.setText("1 x 32 x 32".replace(" ", ""))
    req = dialog.request()
    assert req["input_shape"] == [1, 32, 32] and req["attr"] == "SkipUNet"
    dialog.kind.setCurrentIndex(1)
    assert not dialog.attr.isEnabled()
    dialog.close()
    app.processEvents()
