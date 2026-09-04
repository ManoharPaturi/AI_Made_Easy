"""Shared shape arithmetic for conv/pool blocks (channels-first IR dims)."""
from __future__ import annotations

from ai_made_easy.core.spec import ShapeError, parse_int_list, require_rank


def conv_out(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    out = (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if out <= 0:
        # the effective kernel (with dilation) eats the whole padded input
        eff = dilation * (kernel - 1) + 1
        fit = size + 2 * padding
        raise ShapeError(
            f"kernel {kernel}"
            + (f" (dilated to {eff})" if dilation > 1 else "")
            + f" is too large for a {size}px input with padding {padding} "
            f"(needs ≤ {fit}px): lower kernel_size to ≤ {max(fit, 1)}, "
            f"raise padding to ≥ {(eff - size) // 2 + (1 if (eff - size) % 2 else 0) if eff > size else 0}, "
            f"or reduce the stride"
        )
    return out


def conv_transpose_out(
    size: int, kernel: int, stride: int, padding: int, output_padding: int, dilation: int
) -> int:
    out = (
        (size - 1) * stride
        - 2 * padding
        + dilation * (kernel - 1)
        + output_padding
        + 1
    )
    if out <= 0:
        raise ShapeError(
            f"conv-transpose output dim is non-positive: size={size}, kernel={kernel}, "
            f"stride={stride}, padding={padding}"
        )
    return out


def kernel_tuple(params: dict) -> int:
    """Square kernels are single ints in v1 (keeps params UI-simple)."""
    return int(params["kernel_size"])


def _pos(params: dict, key: str, default: int) -> int:
    val = params.get(key, default)
    return default if val is None else int(val)


def conv_nd(in_shape: list[int], rank: int, params: dict, transpose: bool = False) -> list[int]:
    """Convolution over a [C, *spatial] tensor of the given rank."""
    require_rank(in_shape, rank + 1, "Conv")
    spatial = in_shape[1:]
    k = kernel_tuple(params)
    stride = _pos(params, "stride", 1)
    padding = _pos(params, "padding", 0)
    dilation = _pos(params, "dilation", 1)
    if transpose:
        op = _pos(params, "output_padding", 0)
        out_spatial = [
            conv_transpose_out(s, k, stride, padding, op, dilation) for s in spatial
        ]
    else:
        out_spatial = [conv_out(s, k, stride, padding, dilation) for s in spatial]
    return [int(params["out_channels"]), *out_spatial]


def pool_nd(in_shape: list[int], rank: int, params: dict) -> list[int]:
    require_rank(in_shape, rank + 1, "Pool")
    spatial = in_shape[1:]
    k = kernel_tuple(params)
    stride = _pos(params, "stride", k)
    padding = _pos(params, "padding", 0)
    out_spatial = [conv_out(s, k, stride, padding, 1) for s in spatial]
    return [in_shape[0], *out_spatial]


def parse_target(value: str) -> list[int]:
    """Reshape target: the sample shape, batch always excluded/preserved."""
    dims = parse_int_list(value)
    if any(d <= 0 for d in dims):
        raise ShapeError(
            f"invalid reshape target {value!r}: dims are the sample shape "
            "(batch is preserved automatically); -1 is not supported"
        )
    return dims


def resolve_target(dims: list[int], volume: int) -> list[int]:
    total = 1
    for d in dims:
        total *= d
    if total != volume:
        raise ShapeError(f"reshape target {dims} does not match volume {volume}")
    return list(dims)


def parse_order(value: str, rank: int) -> list[int]:
    order = parse_int_list(value)
    if sorted(order) != list(range(rank)):
        raise ShapeError(
            f"permute order must be a permutation of 0..{rank - 1}, got {order}"
        )
    return order
