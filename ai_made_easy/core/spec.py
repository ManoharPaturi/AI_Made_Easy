"""Block specification primitives.

These dataclasses are the *contract* every block in the library implements.
They are pure Python (no Qt) and JSON-serializable so that the registry can
self-describe for the UI, the CLI, and (later) MCP agents.

Shape convention (canonical IR): channels-first sample shapes, no batch dim.
Images: [C, H, W]; volumes: [C, D, H, W]; sequences: [L, C]; flat: [F].
Keras channels-last translation happens only at codegen time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Param types supported by the property-panel generator and codegen.
PARAM_TYPES = ("int", "float", "bool", "enum", "str")

# A shape function receives the ordered shapes of the block's input ports
# (already resolved, batch dim excluded) plus the block's resolved params,
# and returns the block's output shape. Raise ShapeError on mismatches.
ShapeFn = Callable[[list[list[int]], dict[str, Any]], list[int]]

# A parameter-count function computes the number of trainable parameters of
# a block analytically (no torch needed) from the same inputs.
ParamFn = Callable[[list[list[int]], dict[str, Any]], int]


class ShapeError(ValueError):
    """A block cannot consume the shapes offered by its inputs."""


@dataclass(frozen=True)
class PortSpec:
    """A named input or output connector on a block."""

    name: str
    dtype: str = "tensor"  # tensor | config (config ports carry no shapes)
    multi: bool = False  # accepts more than one incoming wire (input ports)


@dataclass(frozen=True)
class ParamSpec:
    """A single user-editable parameter of a block."""

    name: str
    type: str  # one of PARAM_TYPES
    default: Any = None
    options: tuple[str, ...] = ()  # populated for enum params
    minimum: Any = None
    maximum: Any = None
    help: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "default": self.default,
        }
        if self.options:
            out["options"] = list(self.options)
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        if self.help:
            out["help"] = self.help
        return out


@dataclass(frozen=True)
class BlockDefinition:
    """A block *type* (e.g. "core.dense") — the recipe for canvas instances.

    Codegen fragments. Two layers per framework:
      * ``pytorch_layer`` — an nn.Module constructor line for ``__init__``
        ("" means the block is functional-only).
      * ``pytorch_expr`` — the forward-pass expression. Placeholders:
        ``{self_var}`` = the block's own module attribute, ``{i0}``,``{i1}``
        = the input variable names (port order), then all params plus the
        shape context (``in_features``, ``in_channels``, ``num_features``,
        ``normalized_shape``). Default when empty: ``self.{self_var}({i0})``.
      * ``keras_layer`` — the keras layers constructor (used inline).
      * ``keras_expr`` — the keras call expression; default
        ``{keras_layer}({i0})``. None/"" with ``keras_layer`` also empty
        marks "not supported for Keras export yet".
    """

    type_id: str  # unique, namespaced: "<package>.<name>"
    display_name: str
    category: str
    color: str = "#4a9eff"  # hex node color, per-category theming
    params: tuple[ParamSpec, ...] = ()
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()
    shape_fn: ShapeFn | None = None  # None: not part of tensor flow
    param_fn: ParamFn | None = None  # None: block has no parameters
    # Guardrail: cross-param relations (embed % num_heads, 0..1 ranges...)
    # resolved_params() -> list[(severity, kid-friendly message)]; pure function.
    checks_fn: Any = None
    # Composite blocks (architecture macros): builder(params) -> Fragment.
    # Composites must be expanded into primitives before codegen.
    builder: Any = None
    pytorch_layer: str = ""
    pytorch_expr: str = ""
    keras_layer: str = ""
    keras_expr: str = ""

    def __post_init__(self) -> None:
        if self.params and not isinstance(self.params, tuple):
            raise TypeError(
                f"{self.type_id}: params must be a tuple of ParamSpec, "
                f"got {type(self.params).__name__} (missing trailing comma?)"
            )
        for p in self.params:
            if not isinstance(p, ParamSpec):
                raise TypeError(f"{self.type_id}: param {p!r} is not a ParamSpec")
        for p in (*self.inputs, *self.outputs):
            if not isinstance(p, PortSpec):
                raise TypeError(f"{self.type_id}: port {p!r} is not a PortSpec")

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def to_dict(self) -> dict[str, Any]:
        """Self-describing JSON form — the block's schema for UI/CLI/agents."""
        out = {
            "type_id": self.type_id,
            "display_name": self.display_name,
            "category": self.category,
            "color": self.color,
            "composite": self.builder is not None,
            "params": [p.to_dict() for p in self.params],
            "inputs": [
                {"name": p.name, "dtype": p.dtype, "multi": p.multi} for p in self.inputs
            ],
            "outputs": [{"name": p.name, "dtype": p.dtype} for p in self.outputs],
        }
        return out


def parse_shape(shape_str: str) -> list[int]:
    """Parse a comma-separated shape like '28,28,1' into ints (no batch dim)."""
    parts = [p.strip() for p in shape_str.split(",") if p.strip()]
    if not parts:
        raise ShapeError(f"empty shape: {shape_str!r}")
    dims: list[int] = []
    for p in parts:
        try:
            dims.append(int(p))
        except ValueError:
            raise ShapeError(f"shape dimension {p!r} is not an integer: {shape_str!r}") from None
    if any(d <= 0 for d in dims):
        raise ShapeError(f"shape dims must be positive: {shape_str!r}")
    return dims


def parse_int_list(value: str) -> list[int]:
    """Parse '2,3' style params (kernel sizes, permute orders, targets)."""
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise ShapeError(f"expected comma-separated integers, got {value!r}") from None


def shape_volume(shape: list[int]) -> int:
    vol = 1
    for d in shape:
        vol *= d
    return vol


def require_rank(shape: list[int], rank: int, block: str) -> None:
    if len(shape) != rank:
        names = {1: "[F]", 2: "[L,C]", 3: "[C,H,W]", 4: "[C,D,H,W]"}
        raise ShapeError(
            f"{block} expects rank-{rank} input {names.get(rank, '')}, got {shape}"
        )
