import { type ReactNode, useEffect } from "react";

export function Modal({ title, onClose, children, actions, wide = false }: {
  title: string; onClose: () => void; children: ReactNode; actions?: ReactNode; wide?: boolean;
}) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal${wide ? " wide" : ""}`} role="dialog" aria-label={title}>
        <h3>{title}</h3>
        {children}
        <div className="actions">
          {actions}
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

export function StatusChip({ status }: { status: string }) {
  const cls = status === "finished" || status === "ok" ? "ok"
    : status === "failed" || status === "interrupted" ? "err"
      : status === "running" ? "run" : "";
  return <span className={`chip ${cls}`}>{status}</span>;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${seconds.toFixed(0)} s`;
  const m = Math.floor(seconds / 60);
  return m < 60 ? `${m} min ${Math.round(seconds % 60)} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function metric(value: unknown): string {
  return typeof value === "number" ? Number(value.toPrecision(4)).toString() : "—";
}

/** Loading, error and empty states for a list; renders nothing once there are items. */
export function ListState({ items, error, empty, onRetry, what = "items" }: {
  items: unknown[] | null; error?: string; empty: ReactNode; onRetry?: () => void; what?: string;
}) {
  if (error) {
    return (
      <div className="empty state-error" role="alert">
        Could not load the {what}: {error}
        {onRetry && <div><button onClick={onRetry} style={{ marginTop: 8 }}>Try again</button></div>}
      </div>);
  }
  if (items === null) {
    return <div className="empty" aria-busy="true"><span className="spinner" /> Loading {what}…</div>;
  }
  if (!items.length) return <div className="empty">{empty}</div>;
  return null;
}
