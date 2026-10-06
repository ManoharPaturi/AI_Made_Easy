"""Runtime exports: standalone scripts that serialize the designed model to
ONNX or TorchScript. They embed the exact PyTorch model file the app
generates, so every export format shares one verified model definition.
"""
from __future__ import annotations

from ai_made_easy.core.codegen import emit_graph, generate
from ai_made_easy.core.graph import Graph

_ONNX_MAIN = '''

OUTPUT = Path("{output_path}")


def main() -> None:
    model = build_model().eval()
    dummy = {dummy}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy,), str(OUTPUT),
            input_names=["input"], output_names=["output"],
            dynamic_axes={{"input": {{0: "batch"}}, "output": {{0: "batch"}}}},
            opset_version=18,
        )
    print(f"saved {{OUTPUT}}")
    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(OUTPUT)))
        print("onnx checker: OK")
    except ImportError:
        print("(pip install onnx to validate the exported file)")


if __name__ == "__main__":
    main()
'''

_JIT_MAIN = '''

OUTPUT = Path("{output_path}")


def main() -> None:
    model = build_model().eval()
    dummy = {dummy}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        exported = torch.jit.script(model)
    except Exception:  # noqa: BLE001 — fall back to tracing for dynamic Python
        exported = torch.jit.trace(model, dummy)
    exported.save(str(OUTPUT))
    print(f"saved {{OUTPUT}}")
    reloaded = torch.jit.load(str(OUTPUT))
    with torch.no_grad():
        out = reloaded(dummy)
    print(f"reload check OK — output shape {{tuple(out.shape)}}")


if __name__ == "__main__":
    main()
'''


def _model_source(graph: Graph) -> tuple[str, str]:
    source = generate(graph, "pytorch")
    source = source.split('\nif __name__ == "__main__":')[0].rstrip() + "\n"
    source = source.replace("import torch\n", "from pathlib import Path\n\nimport torch\n", 1)
    plan = emit_graph(graph)
    dims = ", ".join(str(d) for d in plan.input_shape)
    dummy = (f"torch.randint(0, 2, (1, {dims}))" if plan.input_dtype.startswith("int")
             else f"torch.randn(1, {dims})")
    return source, dummy


def generate_onnx_export(graph: Graph, output_path: str = "exports/model.onnx") -> str:
    source, dummy = _model_source(graph)
    return source + _ONNX_MAIN.format(output_path=output_path, dummy=dummy)


def generate_torchscript_export(graph: Graph,
                                output_path: str = "exports/model_torchscript.pt") -> str:
    source, dummy = _model_source(graph)
    return source + _JIT_MAIN.format(output_path=output_path, dummy=dummy)
