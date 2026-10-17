// In-app confirm / prompt dialogs (3.3) instead of the browser's confirm() and prompt():
// they follow the theme, can be read by screen readers and don't block the page.
//   const { confirm, prompt } = useDialogs();
//   if (!(await confirm("Delete 3 runs?", { danger: true, ok: "Delete" }))) return;
import { createContext, type ReactNode, useCallback, useContext, useEffect, useRef, useState } from "react";

type ConfirmOptions = { title?: string; ok?: string; danger?: boolean };
type PromptOptions = { title?: string; ok?: string; label?: string; value?: string;
                       secret?: boolean; placeholder?: string };
type Request =
  | { kind: "confirm"; message: string; options: ConfirmOptions; resolve: (v: boolean) => void }
  | { kind: "prompt"; message: string; options: PromptOptions; resolve: (v: string | null) => void };

interface Dialogs {
  confirm: (message: string, options?: ConfirmOptions) => Promise<boolean>;
  prompt: (message: string, options?: PromptOptions) => Promise<string | null>;
}

const Ctx = createContext<Dialogs | null>(null);

export function useDialogs(): Dialogs {
  const dialogs = useContext(Ctx);
  if (!dialogs) throw new Error("useDialogs outside DialogProvider");
  return dialogs;
}

export function DialogProvider({ children }: { children: ReactNode }) {
  const [queue, setQueue] = useState<Request[]>([]);
  const confirm = useCallback((message: string, options: ConfirmOptions = {}) =>
    new Promise<boolean>((resolve) => setQueue((q) => [...q, { kind: "confirm", message, options, resolve }])), []);
  const prompt = useCallback((message: string, options: PromptOptions = {}) =>
    new Promise<string | null>((resolve) => setQueue((q) => [...q, { kind: "prompt", message, options, resolve }])), []);
  const current = queue[0];
  const done = (value: boolean | string | null) => {
    if (!current) return;
    if (current.kind === "confirm") current.resolve(Boolean(value));
    else current.resolve(typeof value === "string" ? value : null);
    setQueue((q) => q.slice(1));
  };
  const [dialogs] = useState<Dialogs>(() => ({ confirm, prompt }));
  return (
    <Ctx.Provider value={dialogs}>
      {children}
      {current && <AskDialog key={queue.length} request={current} onDone={done} />}
    </Ctx.Provider>
  );
}

function AskDialog({ request, onDone }: { request: Request; onDone: (v: boolean | string | null) => void }) {
  const isPrompt = request.kind === "prompt";
  const opts = request.options as PromptOptions & ConfirmOptions;
  const [value, setValue] = useState(isPrompt ? opts.value ?? "" : "");
  const okRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const cancel = () => onDone(isPrompt ? null : false);
  const ok = () => onDone(isPrompt ? value : true);
  useEffect(() => {
    (isPrompt ? inputRef.current : okRef.current)?.focus();
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); cancel(); }
    };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const title = opts.title ?? (isPrompt ? "Enter a value" : "Are you sure?");
  return (
    <div className="modal-backdrop ask" onMouseDown={(e) => e.target === e.currentTarget && cancel()}>
      <div className="modal narrow" role="alertdialog" aria-modal="true" aria-label={title}
           aria-describedby="ask-message" data-testid={`ask-${request.kind}`}>
        <h3>{title}</h3>
        <p id="ask-message" style={{ whiteSpace: "pre-wrap" }}>{request.message}</p>
        {isPrompt && (
          <form onSubmit={(e) => { e.preventDefault(); ok(); }}>
            <input ref={inputRef} type={opts.secret ? "password" : "text"} value={value}
                   aria-label={opts.label ?? request.message} placeholder={opts.placeholder}
                   onChange={(e) => setValue(e.target.value)} style={{ width: "100%" }}
                   autoComplete="off" />
          </form>)}
        <div className="actions">
          <button onClick={cancel}>Cancel</button>
          <button ref={okRef} className={opts.danger ? "primary danger-fill" : "primary"} onClick={ok}
                  disabled={isPrompt && !value.trim()}>{opts.ok ?? "OK"}</button>
        </div>
      </div>
    </div>
  );
}
