import { type ReactNode, useEffect } from "react";

export function Modal({ title, onClose, children, actions }: {
  title: string; onClose: () => void; children: ReactNode; actions?: ReactNode;
}) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-label={title}>
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
