// Where runs train: the "Run on" selector (shared by training, sweeps and pipelines)
// and the dialog that adds, tests and removes backends. Sign-in stays with each tool
// (~/.ssh, ~/.kaggle/kaggle.json, modal token, Google Drive for desktop): no field here
// ever takes a password, token or key.
import { useCallback, useEffect, useState, useSyncExternalStore } from "react";

import { api } from "../api";
import { useStore } from "../state";
import { useDialogs } from "./Dialogs";
import type { BackendInfo, BackendList, BackendTest } from "../types";
import { ListState, Modal } from "./Modal";

const KEY = "aime.runOn";
const listeners = new Set<() => void>();

function readRunOn(): string {
  return localStorage.getItem(KEY) || "local";
}

/** The chosen backend, remembered across pages and reloads. */
export function useRunOn(): [string, (on: string) => void] {
  const value = useSyncExternalStore((cb) => {
    listeners.add(cb);
    return () => listeners.delete(cb);
  }, readRunOn);
  const set = useCallback((on: string) => {
    localStorage.setItem(KEY, on);
    listeners.forEach((cb) => cb());
  }, []);
  return [value, set];
}

const KIND_LABELS: Record<string, string> = {
  local: "this computer", ssh: "SSH", kaggle: "Kaggle", colab: "Colab", modal: "Modal",
  container: "Docker",
};

function label(b: BackendInfo): string {
  if (b.kind === "local") return "This computer";
  const kind = KIND_LABELS[b.kind] ?? b.kind;
  return b.name === b.kind ? kind : `${b.name} (${kind})`;
}

/** Remote runs whose curves do not stream say so next to the selector. */
export function RunOnSelect({ testId = "run-on" }: { testId?: string }) {
  const [on, setOn] = useRunOn();
  const [list, setList] = useState<BackendInfo[]>([]);
  const [managing, setManaging] = useState(false);
  const refresh = useCallback(() => {
    api.backends().then((r) => setList(r.backends)).catch(() => setList([]));
  }, []);
  useEffect(refresh, [refresh]);
  const current = list.find((b) => b.name === on);
  return (
    <>
      <label className="dim" title="Where the run trains">Run on{" "}
        <select data-testid={testId} value={current ? on : "local"} onChange={(e) => {
          if (e.target.value === "__manage") setManaging(true); else setOn(e.target.value);
        }}>
          {(list.length ? list : [{ name: "local", kind: "local" } as BackendInfo]).map((b) => (
            <option key={b.name} value={b.name}>{label(b)}</option>))}
          <option value="__manage">Manage backends…</option>
        </select>
      </label>
      {current && !current.live && (
        <span className="muted" data-testid="run-on-note">curves arrive when the run reports back</span>)}
      {managing && <BackendsModal onClose={() => { setManaging(false); refresh(); }} />}
    </>
  );
}

function parseOption(text: string): unknown {
  const t = text.trim();
  if (t === "") return undefined;
  try {
    return JSON.parse(t);  // true / 30 / ["owner/dataset"]
  } catch {
    return t;
  }
}

export function BackendsModal({ onClose }: { onClose: () => void }) {
  const { notify } = useStore();
  const { confirm } = useDialogs();
  const [data, setData] = useState<BackendList | null>(null);
  const [error, setError] = useState("");
  const [tests, setTests] = useState<Record<string, BackendTest | "testing">>({});
  const [name, setName] = useState("");
  const [kind, setKind] = useState("ssh");
  const [options, setOptions] = useState<Record<string, string>>({});
  const refresh = useCallback(() => {
    api.backends().then((d) => { setData(d); setError(""); })
      .catch((e) => setError(String(e.message)));
  }, [notify]);
  useEffect(refresh, [refresh]);

  const test = async (on: string) => {
    setTests((t) => ({ ...t, [on]: "testing" }));
    try {
      const result = await api.testBackend(on);
      setTests((t) => ({ ...t, [on]: result }));
    } catch (e) {
      setTests((t) => ({ ...t, [on]: { ok: false, message: String((e as Error).message),
                                        backend: on, kind: "", accelerator: {} } }));
    }
  };
  const save = async () => {
    const parsed: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(options)) {
      const value = parseOption(v);
      if (value !== undefined) parsed[k] = value;
    }
    try {
      await api.saveBackend(name.trim(), kind, parsed);
      notify(`Saved backend ${name}`);
      setName("");
      setOptions({});
      refresh();
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const remove = async (n: string) => {
    if (!(await confirm(`Remove the backend ${n}? Its runs stay in the history.`,
                        { title: "Remove backend", ok: "Remove", danger: true }))) return;
    await api.deleteBackend(n).catch((e) => notify(String(e.message), true));
    refresh();
  };
  const kindOptions = data?.options[kind] ?? {};
  const saved = new Set(["local", "kaggle", "colab", "modal", "container"]);
  return (
    <Modal title="Backends: where runs train" onClose={onClose} wide>
      <p className="dim">Sign-in stays with each tool: <code>~/.ssh/config</code> and keys,
        <code> ~/.kaggle/kaggle.json</code>, <code>modal token new</code>, Google Drive for
        desktop. AI Made Easy never asks for or stores a password, token or key.</p>
      <table data-testid="backends">
        <thead><tr><th>Name</th><th>Kind</th><th>Options</th><th>Check</th><th /></tr></thead>
        <tbody>
          {(data?.backends ?? []).map((b) => {
            const t = tests[b.name];
            return (
              <tr key={b.name}>
                <td><b>{b.name}</b></td>
                <td>{KIND_LABELS[b.kind] ?? b.kind}{!b.live && <span className="muted"> · per poll</span>}</td>
                <td className="dim">{Object.entries(b.options).map(([k, v]) =>
                  `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(" ")}</td>
                <td>
                  <button onClick={() => test(b.name)} disabled={t === "testing"}>Test</button>{" "}
                  {t && t !== "testing" && (
                    <span className={t.ok ? "" : "sev-error"} data-testid={`backend-test-${b.name}`}>
                      {t.ok ? "✓ " : "✗ "}{t.message}
                      {t.accelerator?.gpu && ` · ${t.accelerator.gpu}`}
                      {t.accelerator?.device && <span className="muted"> · budget: {t.accelerator.device}</span>}
                    </span>)}
                  {t === "testing" && <span className="muted">checking…</span>}
                </td>
                <td>{!saved.has(b.name) &&
                  <button className="danger" onClick={() => remove(b.name)}>Remove</button>}</td>
              </tr>);
          })}
        </tbody>
      </table>
      <ListState items={data?.backends ?? null} error={error} onRetry={refresh} what="backends"
                 empty="No backends." />
      <h3>Add a backend</h3>
      <div className="form">
        <div className="field"><label htmlFor="backend-name">Name</label>
          <input id="backend-name" value={name} placeholder="gpu-box"
                 onChange={(e) => setName(e.target.value)} /></div>
        <div className="field"><label htmlFor="backend-kind">Kind</label>
          <select id="backend-kind" value={kind} onChange={(e) => { setKind(e.target.value); setOptions({}); }}>
            {(data?.kinds ?? ["ssh"]).filter((k) => k !== "local").map((k) => (
              <option key={k} value={k}>{KIND_LABELS[k] ?? k}</option>))}
          </select></div>
        {Object.entries(kindOptions).map(([opt, help]) => (
          <div className="field" key={opt}><label htmlFor={`backend-opt-${opt}`}>{opt}</label>
            <input id={`backend-opt-${opt}`} value={options[opt] ?? ""} placeholder={help}
                   title={help} onChange={(e) => setOptions((o) => ({ ...o, [opt]: e.target.value }))} /></div>))}
        <div><button className="primary" disabled={!name.trim()} onClick={save}
                     data-testid="save-backend">Save backend</button></div>
      </div>
    </Modal>
  );
}
