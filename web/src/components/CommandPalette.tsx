// Command palette (Ctrl/⌘ K) and find-a-block (Ctrl/⌘ F), 3.3. Arrow keys move, Enter runs,
// Escape closes. Typing ">" in find mode switches to commands and "@" in commands to blocks.
import { useEffect, useMemo, useRef, useState } from "react";

import { type ActionDef, keyLabel, matchesQuery } from "../actions";
import { useStore } from "../state";

export type PaletteMode = "commands" | "find";
type Item = { key: string; label: string; detail?: string; keys?: string; swatch?: string;
              group: string; run: () => void };

export function CommandPalette({ mode, actions, run, onClose }: {
  mode: PaletteMode;
  actions: ActionDef[];
  run: (id: string) => void;
  onClose: () => void;
}) {
  const { nodes, blockMap, canvas } = useStore();
  const [query, setQuery] = useState(mode === "find" ? "@" : "");
  const [index, setIndex] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);
  useEffect(() => input.current?.focus(), []);

  const finding = query.startsWith("@");
  const text = finding ? query.slice(1) : query.replace(/^>/, "");
  const items: Item[] = useMemo(() => {
    if (finding) {
      return nodes
        .map((n) => {
          const def = blockMap.get(n.data.typeId);
          return { key: n.id, label: def?.display_name ?? n.data.typeId, detail: n.id,
                   swatch: def?.color, group: def?.category ?? "Blocks",
                   run: () => canvas.current?.focus(n.id) };
        })
        .filter((i) => matchesQuery(`${i.label} ${i.detail} ${i.group}`, text));
    }
    return actions
      .filter((a) => matchesQuery(`${a.label} ${a.group} ${a.hint ?? ""} ${a.id}`, text))
      .map((a) => ({ key: a.id, label: a.label, keys: a.keys, group: a.group, run: () => run(a.id) }));
  }, [finding, text, nodes, blockMap, canvas, actions, run]);

  useEffect(() => setIndex(0), [query]);
  useEffect(() => {
    list.current?.querySelector(".palette-item.active")?.scrollIntoView({ block: "nearest" });
  }, [index]);

  const choose = (item: Item | undefined) => {
    if (!item) return;
    onClose();
    window.setTimeout(item.run, 0);   // after the palette is gone (focus, dialogs)
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setIndex((i) => Math.min(i + 1, items.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setIndex((i) => Math.max(i - 1, 0)); }
    else if (e.key === "Enter") { e.preventDefault(); choose(items[index]); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); onClose(); }
  };
  const label = finding ? "Find a block" : "Command palette";
  return (
    <div className="modal-backdrop palette-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="palette" role="dialog" aria-label={label} data-testid="palette">
        <input ref={input} value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={onKey}
               placeholder={finding ? "Block name or id…  (> for commands)" : "Type a command…  (@ finds a block)"}
               aria-label={label} role="combobox" aria-expanded="true" aria-controls="palette-list"
               aria-activedescendant={items[index] ? `palette-${items[index].key}` : undefined} />
        <div className="palette-list" id="palette-list" role="listbox" ref={list}>
          {items.map((item, i) => (
            <div key={item.key} id={`palette-${item.key}`} role="option" aria-selected={i === index}
                 className={`palette-item ${i === index ? "active" : ""}`}
                 onMouseMove={() => setIndex(i)} onClick={() => choose(item)}>
              {item.swatch && <span className="swatch" style={{ background: item.swatch }} />}
              <span className="palette-label">{item.label}</span>
              {item.detail && <span className="muted">{item.detail}</span>}
              <span className="spacer" />
              <span className="muted palette-group">{item.group}</span>
              {item.keys && <kbd>{keyLabel(item.keys)}</kbd>}
            </div>))}
          {!items.length && <div className="empty">{finding
            ? (nodes.length ? `No block matches “${text}”.` : "The canvas is empty.")
            : `No command matches “${text}”.`}</div>}
        </div>
      </div>
    </div>
  );
}

export function ShortcutsSheet({ actions, onClose }: { actions: ActionDef[]; onClose: () => void }) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);
  const keyed = actions.filter((a) => a.keys);
  const groups = [...new Set(keyed.map((a) => a.group))];
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal wide" role="dialog" aria-label="Keyboard shortcuts" data-testid="shortcuts">
        <h3>Keyboard shortcuts</h3>
        <div className="shortcut-groups">
          {groups.map((g) => (
            <section key={g}>
              <h4>{g}</h4>
              <table><tbody>
                {keyed.filter((a) => a.group === g).map((a) => (
                  <tr key={a.id}><td>{a.label}</td><td className="num"><kbd>{keyLabel(a.keys!)}</kbd></td></tr>))}
              </tbody></table>
            </section>))}
        </div>
        <p className="muted">Also: drag from the library or press Enter on a block there to add it; drag
          wires between ports; Shift-drag to select a box of blocks.</p>
        <div className="actions"><button onClick={onClose}>Close</button></div>
      </div>
    </div>
  );
}
