"""Core engine: registry, blocks, IR, codegen, runner, persistence.

PURE PYTHON — no Qt imports allowed in this package. Every capability is a
plain function over JSON-serializable data so the UI, CLI, and future MCP
agents all drive the exact same engine.
"""
