import { useMemo, useState } from "react";

import { useStore } from "../state";
import type { BlockDef } from "../types";

export const DRAG_TYPE = "application/x-aime-block";

function matches(def: BlockDef, query: string): boolean {
  const q = query.toLowerCase();
  return def.display_name.toLowerCase().includes(q) || def.type_id.toLowerCase().includes(q)
    || def.category.toLowerCase().includes(q) || def.description.toLowerCase().includes(q);
}

export function Library({ onAdd }: { onAdd: (typeId: string) => void }) {
  const { blocks } = useStore();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>({ "Input / Output": true });

  const groups = useMemo(() => {
    const out = new Map<string, BlockDef[]>();
    for (const b of blocks) {
      if (query && !matches(b, query)) continue;
      if (!out.has(b.category)) out.set(b.category, []);
      out.get(b.category)!.push(b);
    }
    return [...out.entries()];
  }, [blocks, query]);

  return (
    <aside className="panel library">
      <div className="panel-title">Block Library</div>
      <input className="search" placeholder={`Search ${blocks.length} blocks`} value={query}
             onChange={(e) => setQuery(e.target.value)} aria-label="Search blocks" />
      <div className="library-list">
        {groups.map(([category, items]) => {
          const expanded = Boolean(query) || open[category];
          return (
            <div className="category" key={category}>
              <button onClick={() => setOpen((o) => ({ ...o, [category]: !o[category] }))}>
                {expanded ? "▾" : "▸"} {category} <span className="muted">{items.length}</span>
              </button>
              {expanded && items.map((b) => (
                <div key={b.type_id} className="block-item" draggable title={b.description}
                     data-testid={`lib-${b.type_id}`}
                     onDragStart={(e) => {
                       e.dataTransfer.setData(DRAG_TYPE, b.type_id);
                       e.dataTransfer.effectAllowed = "move";
                     }}
                     onDoubleClick={() => onAdd(b.type_id)}>
                  <span className="swatch" style={{ background: b.color }} />
                  {b.display_name}
                  {b.missing?.length ? (
                    <span className="chip" title={`Requires ${b.missing.join(", ")} — pip install ${
                      b.extra ? `'ai-made-easy[${b.extra}]'` : b.missing.join(" ")}`}>needs {b.missing[0]}</span>
                  ) : null}
                </div>
              ))}
            </div>
          );
        })}
        {!groups.length && <div className="empty">No blocks match “{query}”.</div>}
      </div>
    </aside>
  );
}
