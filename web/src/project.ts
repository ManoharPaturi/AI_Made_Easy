// Project JSON <-> React Flow nodes / edges.
import type { Edge, Node } from "@xyflow/react";

import type { BlockDef, Project } from "./types";

export type BlockData = { typeId: string; params: Record<string, unknown> };
export type FlowNode = Node<BlockData, "block">;

export function edgeId(source: string, sourceHandle: string, target: string, targetHandle: string) {
  return `${source}/${sourceHandle}->${target}/${targetHandle}`;
}

export function fromProject(project: Project): { nodes: FlowNode[]; edges: Edge[] } {
  const nodes: FlowNode[] = project.nodes.map((n) => ({
    id: n.id,
    type: "block",
    position: { x: n.position?.[0] ?? 0, y: n.position?.[1] ?? 0 },
    data: { typeId: n.type, params: { ...(n.params ?? {}) } },
  }));
  const edges: Edge[] = project.edges.map((e) => {
    const [source, sourceHandle = "out"] = e.from.split("/");
    const [target, targetHandle = "in"] = e.to.split("/");
    return { id: edgeId(source, sourceHandle, target, targetHandle), source, sourceHandle,
             target, targetHandle };
  });
  return { nodes, edges };
}

export function toProject(name: string, nodes: FlowNode[], edges: Edge[],
                          meta: Record<string, unknown> = {}, schemaVersion = 2): Project {
  return {
    schema_version: schemaVersion,
    name: name || "untitled",
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.typeId,
      params: n.data.params,
      position: [Math.round(n.position.x), Math.round(n.position.y)],
    })),
    edges: edges.map((e) => ({
      from: `${e.source}/${e.sourceHandle ?? "out"}`,
      to: `${e.target}/${e.targetHandle ?? "in"}`,
    })),
    meta,
  };
}

/** Only what affects validation (not positions / selection). */
export function structureKey(nodes: FlowNode[], edges: Edge[]): string {
  return JSON.stringify([
    nodes.map((n) => [n.id, n.data.typeId, n.data.params]),
    edges.map((e) => e.id),
  ]);
}

export function defaultParams(def: BlockDef): Record<string, unknown> {
  return Object.fromEntries(def.params.map((p) => [p.name, p.default]));
}

export function newNodeId(typeId: string, taken: Set<string>): string {
  const base = typeId.split(".").pop()!.replace(/[^a-z0-9]+/gi, "_").toLowerCase() || "block";
  for (let i = 1; ; i++) {
    const id = i === 1 ? base : `${base}_${i}`;
    if (!taken.has(id)) return id;
  }
}

export function paramSummary(def: BlockDef | undefined, params: Record<string, unknown>): string {
  if (!def) return "";
  return def.params
    .slice(0, 3)
    .map((p) => `${p.name} ${formatValue(params[p.name] ?? p.default)}`)
    .join(" · ");
}

export function formatValue(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(4).replace(/\.?0+$/, "");
  if (typeof value === "boolean") return value ? "on" : "off";
  return String(value ?? "");
}

export function shapeLabel(shape: number[] | undefined): string {
  return shape ? `[${shape.join(", ")}]` : "";
}

const NODE_W = 230;
const NODE_H = 56;

/** True when any two nodes overlap (e.g. projects saved without positions). */
export function hasOverlaps(nodes: FlowNode[]): boolean {
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i].position;
      const b = nodes[j].position;
      if (Math.abs(a.x - b.x) < NODE_W - 10 && Math.abs(a.y - b.y) < NODE_H - 6) return true;
    }
  }
  return false;
}

/** Layered left-to-right layout by data flow; unconnected blocks in rows below. */
export function autoLayout(nodes: FlowNode[], edges: Edge[]): FlowNode[] {
  const wired = new Set(edges.flatMap((e) => [e.source, e.target]));
  const incoming = new Map<string, string[]>();
  for (const e of edges) incoming.set(e.target, [...(incoming.get(e.target) ?? []), e.source]);
  const layer = new Map<string, number>();
  const depth = (id: string, seen: Set<string>): number => {
    if (layer.has(id)) return layer.get(id)!;
    if (seen.has(id)) return 0; // cycle guard
    seen.add(id);
    const parents = incoming.get(id) ?? [];
    const d = parents.length ? Math.max(...parents.map((p) => depth(p, seen))) + 1 : 0;
    layer.set(id, d);
    return d;
  };
  const rows = new Map<number, number>();
  const placed = new Map<string, { x: number; y: number }>();
  for (const n of nodes.filter((n) => wired.has(n.id))) {
    const d = depth(n.id, new Set());
    const row = rows.get(d) ?? 0;
    rows.set(d, row + 1);
    placed.set(n.id, { x: d * (NODE_W + 50), y: row * (NODE_H + 44) });
  }
  const bottom = Math.max(0, ...[...placed.values()].map((p) => p.y)) + NODE_H + 100;
  const perRow = Math.max(4, Math.min(8, Math.max(...rows.keys(), 0) + 1));
  nodes.filter((n) => !wired.has(n.id)).forEach((n, i) => {
    placed.set(n.id, { x: (i % perRow) * (NODE_W + 30), y: bottom + Math.floor(i / perRow) * (NODE_H + 34) });
  });
  return nodes.map((n) => ({ ...n, position: placed.get(n.id) ?? n.position }));
}
