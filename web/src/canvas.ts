// Canvas conveniences (3.3): copy / paste of blocks, notes and groups.
// Mirrors ai_made_easy/core/canvas.py, so a selection copied in one UI pastes in the other.
import type { Project } from "./types";

export const MARKER = "aime_clipboard";
export const GROUP_COLORS = ["#91A7FF", "#63E6BE", "#FFC078", "#F783AC", "#B197FC", "#99E9F2"];

export type Note = { id: string; text: string; position: [number, number]; size: [number, number] };
export type Group = { id: string; title: string; nodes: string[]; color: string };
type PNode = Project["nodes"][number];
type PEdge = Project["edges"][number];
export type Fragment = { [MARKER]: 1; nodes: PNode[]; edges: PEdge[]; groups: Group[] };

const ends = (e: PEdge) => {
  const [src, srcPort] = splitLast(e.from);
  const [dst, dstPort] = splitLast(e.to);
  return { src, srcPort, dst, dstPort };
};

function splitLast(ref: string): [string, string] {
  const i = ref.lastIndexOf("/");
  return [ref.slice(0, i), ref.slice(i + 1)];
}

export function notesOf(meta: Record<string, unknown>): Note[] {
  return Array.isArray(meta.notes) ? (meta.notes as Note[]) : [];
}

export function groupsOf(meta: Record<string, unknown>): Group[] {
  return Array.isArray(meta.groups) ? (meta.groups as Group[]) : [];
}

export function copyFragment(design: Project, ids: Iterable<string>): Fragment {
  const chosen = new Set(ids);
  return {
    [MARKER]: 1,
    nodes: design.nodes.filter((n) => chosen.has(n.id)).map((n) => structuredClone(n)),
    edges: design.edges.filter((e) => {
      const { src, dst } = ends(e);
      return chosen.has(src) && chosen.has(dst);
    }),
    groups: groupsOf(design.meta ?? {}).filter((g) => g.nodes.length
      && g.nodes.every((n) => chosen.has(n))),
  };
}

/** A clipboard fragment or a whole design from clipboard text, else null. */
export function parseClipboard(text: string): Fragment | null {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return null;
  }
  if (!data || typeof data !== "object") return null;
  const d = data as Record<string, unknown>;
  if (!Array.isArray(d.nodes)) return null;
  const nodes = d.nodes as PNode[];
  if (!nodes.every((n) => n && typeof n === "object" && "id" in n && "type" in n)) return null;
  const edges = (Array.isArray(d.edges) ? d.edges : []) as PEdge[];
  if (!edges.every((e) => String(e?.from ?? "").includes("/") && String(e?.to ?? "").includes("/"))) {
    return null;
  }
  const groups = (d.groups ?? (d.meta as Record<string, unknown> | undefined)?.groups ?? []) as Group[];
  return { [MARKER]: 1, nodes, edges, groups: Array.isArray(groups) ? groups : [] };
}

function fresh(base: string, taken: Set<string>): string {
  if (!taken.has(base)) return base;
  const stem = base.replace(/_\d+$/, "") || "block";
  let i = 2;
  while (taken.has(`${stem}_${i}`)) i++;
  return `${stem}_${i}`;
}

function nextId(items: { id: string }[], prefix: string): string {
  const taken = new Set(items.map((i) => i.id));
  let n = 1;
  while (taken.has(`${prefix}${n}`)) n++;
  return `${prefix}${n}`;
}

export function addGroup(meta: Record<string, unknown>, nodeIds: string[], title = "Group",
                         color?: string): { meta: Record<string, unknown>; id: string } {
  const groups = groupsOf(meta);
  const id = nextId(groups, "group");
  const group = { id, title: title || "Group", nodes: [...new Set(nodeIds)],
                  color: color ?? GROUP_COLORS[(groups.length + 1) % GROUP_COLORS.length] };
  return { meta: { ...meta, groups: [...groups, group] }, id };
}

export function addNote(meta: Record<string, unknown>, text: string,
                        position: [number, number]): { meta: Record<string, unknown>; id: string } {
  const notes = notesOf(meta);
  const id = nextId(notes, "note");
  const note: Note = { id, text, position, size: [220, 120] };
  return { meta: { ...meta, notes: [...notes, note] }, id };
}

/** Paste a fragment into a design; returns the new design and the pasted ids. */
export function pasteFragment(design: Project, fragment: Fragment,
                              offset: [number, number] = [40, 40],
                              at?: [number, number]): { design: Project; ids: string[] } {
  const taken = new Set(design.nodes.map((n) => n.id));
  const pos = fragment.nodes.map((n) => n.position ?? [0, 0]);
  const [dx, dy] = at && pos.length
    ? [at[0] - Math.min(...pos.map((p) => p[0])), at[1] - Math.min(...pos.map((p) => p[1]))]
    : offset;
  const renamed = new Map<string, string>();
  const nodes = [...design.nodes];
  for (const n of fragment.nodes) {
    const id = fresh(n.id, taken);
    taken.add(id);
    renamed.set(n.id, id);
    const p = n.position ?? [0, 0];
    nodes.push({ id, type: n.type, params: structuredClone(n.params ?? {}),
                 position: [p[0] + dx, p[1] + dy] });
  }
  const edges = [...design.edges];
  for (const e of fragment.edges) {
    const { src, srcPort, dst, dstPort } = ends(e);
    if (renamed.has(src) && renamed.has(dst)) {
      edges.push({ from: `${renamed.get(src)}/${srcPort}`, to: `${renamed.get(dst)}/${dstPort}` });
    }
  }
  let meta = { ...(design.meta ?? {}) } as Record<string, unknown>;
  for (const g of fragment.groups ?? []) {
    const members = g.nodes.filter((n) => renamed.has(n)).map((n) => renamed.get(n)!);
    if (members.length) meta = addGroup(meta, members, g.title, g.color).meta;
  }
  return { design: { ...design, nodes, edges, meta }, ids: [...renamed.values()] };
}

/** Groups lose deleted blocks; empty groups disappear. */
export function cleanMeta(meta: Record<string, unknown>, alive: Set<string>): Record<string, unknown> {
  const out = { ...meta };
  const groups = groupsOf(meta).map((g) => ({ ...g, nodes: g.nodes.filter((n) => alive.has(n)) }))
    .filter((g) => g.nodes.length);
  if (groups.length) out.groups = groups; else delete out.groups;
  if (!notesOf(meta).length) delete out.notes;
  return out;
}

export function groupBounds(group: Group, positions: Map<string, { x: number; y: number }>,
                            size: [number, number] = [230, 60], pad = 24) {
  const pts = group.nodes.map((n) => positions.get(n)).filter(Boolean) as { x: number; y: number }[];
  if (!pts.length) return null;
  const x = Math.min(...pts.map((p) => p.x)) - pad;
  const y = Math.min(...pts.map((p) => p.y)) - pad - 22;
  const x1 = Math.max(...pts.map((p) => p.x)) + size[0] + pad;
  const y1 = Math.max(...pts.map((p) => p.y)) + size[1] + pad;
  return { x, y, width: x1 - x, height: y1 - y };
}
