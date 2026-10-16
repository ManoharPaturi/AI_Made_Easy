import { useStore } from "../state";

/** The toast queue (3.3): newest at the bottom, errors stay longer, × closes one. */
export function Toasts() {
  const { toasts, dismissToast } = useStore();
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.error ? "error" : ""}`} data-testid="toast">
          <span>{t.text}{t.count > 1 && <span className="muted"> ×{t.count}</span>}</span>
          <button className="icon" aria-label="Dismiss" onClick={() => dismissToast(t.id)}>×</button>
        </div>))}
    </div>
  );
}
