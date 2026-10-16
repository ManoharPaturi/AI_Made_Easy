// The File menu (3.3): replaces the row of New / Open / Save / Download / Import buttons.
import { useEffect, useRef, useState } from "react";

import { ACTION_MAP, keyLabel } from "../actions";
import { recentProjects } from "../recent";

const ITEMS = ["file.new", "file.open", "-", "recent", "-", "file.save", "file.download", "-",
               "file.import_model", "file.export_png"];
const TEST_IDS: Record<string, string> = { "file.new": "new-project", "file.open": "open-project" };

export function FileMenu({ run, openRecent }: { run: (id: string) => void; openRecent: (name: string) => void }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const first = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    first.current?.focus();
    const away = (e: MouseEvent) => { if (!root.current?.contains(e.target as Node)) setOpen(false); };
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("mousedown", away);
    window.addEventListener("keydown", key);
    return () => { window.removeEventListener("mousedown", away); window.removeEventListener("keydown", key); };
  }, [open]);
  const recent = open ? recentProjects() : [];
  const pick = (fn: () => void) => { setOpen(false); fn(); };
  // arrow keys move between the menu items
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const items = [...(root.current?.querySelectorAll<HTMLButtonElement>("[role=menuitem]") ?? [])];
    const i = items.indexOf(document.activeElement as HTMLButtonElement);
    items[(i + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length]?.focus();
  };
  let firstSet = false;
  return (
    <div className="menu" ref={root}>
      <button aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}
              data-testid="file-menu">File ▾</button>
      {open && (
        <div className="menu-list" role="menu" aria-label="File" onKeyDown={onKey}>
          {ITEMS.map((id, i) => {
            if (id === "-") return <div key={i} className="menu-sep" role="separator" />;
            if (id === "recent") {
              return (
                <div key="recent" role="group" aria-label="Recent projects">
                  <div className="menu-heading">Recent</div>
                  {recent.length ? recent.map((name) => (
                    <button key={name} role="menuitem" className="menu-item"
                            onClick={() => pick(() => openRecent(name))}>{name}</button>))
                    : <div className="menu-heading muted">Nothing yet</div>}
                </div>);
            }
            const a = ACTION_MAP.get(id)!;
            const ref = firstSet ? undefined : first;
            firstSet = true;
            return (
              <button key={id} ref={ref} role="menuitem" className="menu-item" data-testid={TEST_IDS[id]}
                      onClick={() => pick(() => run(id))}>
                <span>{a.label}</span>{a.keys && <kbd>{keyLabel(a.keys)}</kbd>}
              </button>);
          })}
        </div>)}
    </div>
  );
}
