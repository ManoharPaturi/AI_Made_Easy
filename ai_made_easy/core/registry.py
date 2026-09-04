"""Block registry: the catalog of every block type the app knows about.

The registry is intentionally boring and introspectable — ``list_blocks()``
returns full JSON schemas so any frontend (canvas UI, CLI, future MCP agent)
can discover blocks without importing Qt or reading source.
"""
from __future__ import annotations

from ai_made_easy.core.spec import BlockDefinition


class RegistryError(KeyError):
    pass


class BlockRegistry:
    def __init__(self) -> None:
        self._blocks: dict[str, BlockDefinition] = {}

    def register(self, block: BlockDefinition) -> BlockDefinition:
        if block.type_id in self._blocks:
            raise RegistryError(f"duplicate block type_id: {block.type_id}")
        self._blocks[block.type_id] = block
        return block

    def get(self, type_id: str) -> BlockDefinition:
        try:
            return self._blocks[type_id]
        except KeyError:
            raise RegistryError(
                f"unknown block type: {type_id!r} (known: {sorted(self._blocks)})"
            ) from None

    def has(self, type_id: str) -> bool:
        return type_id in self._blocks

    def all(self) -> list[BlockDefinition]:
        return list(self._blocks.values())

    def by_category(self) -> dict[str, list[BlockDefinition]]:
        out: dict[str, list[BlockDefinition]] = {}
        for b in self._blocks.values():
            out.setdefault(b.category, []).append(b)
        return out

    def list_blocks(self) -> list[dict]:
        """JSON-schema view of every registered block (agent-friendly)."""
        return [b.to_dict() for b in self._blocks.values()]

    def search(self, query: str, limit: int = 30) -> list[BlockDefinition]:
        """Case-insensitive substring search over display name, type_id and
        category — the palette search + in-canvas chooser both use this."""
        q = query.strip().lower()
        if not q:
            return []
        hits = [
            b for b in self._blocks.values()
            if q in b.display_name.lower() or q in b.type_id.lower()
            or q in b.category.lower()
        ]
        hits.sort(key=lambda b: (0 if b.display_name.lower().startswith(q) else 1,
                                 b.category, b.display_name))
        return hits[:limit]


_REGISTRY: BlockRegistry | None = None


def get_registry() -> BlockRegistry:
    """Global registry; imports the built-in block library on first use."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = BlockRegistry()
        import ai_made_easy.core.blocks  # noqa: F401  (registers built-ins)
    return _REGISTRY
