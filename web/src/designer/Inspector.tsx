import { useEffect, useState } from "react";

import { api } from "../api";
import { CodeView } from "../components/CodeView";
import { formatValue, shapeLabel } from "../project";
import { useStore } from "../state";
import type { Budget, BudgetReport, DeviceProfile, ParamSpec, Summary } from "../types";

function ParamField({ spec, value, onChange }: {
  spec: ParamSpec; value: unknown; onChange: (v: unknown) => void;
}) {
  const [text, setText] = useState(formatValue(value));
  useEffect(() => setText(value === undefined ? "" : String(value)), [value]);
  const id = `param-${spec.name}`;
  let input;
  if (spec.type === "bool") {
    input = <input id={id} type="checkbox" checked={Boolean(value)}
                   onChange={(e) => onChange(e.target.checked)} />;
  } else if (spec.type === "enum") {
    input = (
      <select id={id} value={String(value)} onChange={(e) => onChange(e.target.value)}>
        {(spec.options ?? []).map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    );
  } else {
    const numeric = spec.type === "int" || spec.type === "float";
    const commit = () => {
      if (!numeric) return onChange(text);
      const n = spec.type === "int" ? parseInt(text, 10) : parseFloat(text);
      if (Number.isNaN(n)) return setText(String(value));
      onChange(n);
    };
    input = (
      <input id={id} value={text} inputMode={numeric ? "decimal" : undefined}
             onChange={(e) => setText(e.target.value)} onBlur={commit}
             onKeyDown={(e) => e.key === "Enter" && commit()} />
    );
  }
  const range = spec.minimum != null || spec.maximum != null
    ? ` (${spec.minimum ?? "−∞"} … ${spec.maximum ?? "∞"})` : "";
  return (
    <div className="field">
      <label htmlFor={id}>{spec.name}<span className="muted">{range}</span></label>
      {input}
      {spec.help && <div className="help">{spec.help}</div>}
    </div>
  );
}

function Properties() {
  const { selected, nodes, blockMap, setParam, commit, validation, dataIssues, name, setName,
          blocks } = useStore();
  const node = nodes.find((n) => n.id === selected);
  if (!node) {
    return (
      <div className="form">
        <div className="field">
          <label htmlFor="project-name">Project name</label>
          <input id="project-name" value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <p className="muted">
          {nodes.length} blocks on the canvas · {blocks.length} in the library. Select a block to
          edit its parameters; drag blocks from the library or double-click them to add.
        </p>
      </div>
    );
  }
  const def = blockMap.get(node.data.typeId);
  const issues = validation?.issues.filter((i) => i.node === node.id) ?? [];
  const data = dataIssues.filter((i) => i.node_id === node.id);
  return (
    <div className="form">
      <div>
        <strong>{def?.display_name ?? node.data.typeId}</strong>
        <div className="muted">{node.id} · {node.data.typeId}
          {validation?.shapes[node.id] && <> · output {shapeLabel(validation.shapes[node.id])}</>}
        </div>
        {def?.description && <p className="dim">{def.description}</p>}
      </div>
      {[...issues.map((i) => [i.severity, i.message]), ...data.map((i) => [i.severity, i.message])]
        .map(([sev, msg], k) => <div key={k} className={`sev-${sev}`}>{sev === "error" ? "✕" : "⚠"} {msg}</div>)}
      {(def?.params ?? []).map((spec) => (
        <ParamField key={spec.name} spec={spec} value={node.data.params[spec.name] ?? spec.default}
                    onChange={(v) => { commit(); setParam(node.id, spec.name, v); }} />
      ))}
      {!def?.params.length && <p className="muted">This block has no parameters.</p>}
    </div>
  );
}

const LIMITS: { key: keyof Budget; label: string; unit: string }[] = [
  { key: "max_train_memory_gb", label: "Max training memory", unit: "GB" },
  { key: "max_latency_ms", label: "Max latency", unit: "ms" },
  { key: "max_params_m", label: "Max parameters", unit: "M" },
  { key: "max_model_mb", label: "Max model size", unit: "MB" },
];

function flops(n: number): string {
  for (const [unit, size] of [["T", 1e12], ["G", 1e9], ["M", 1e6], ["K", 1e3]] as const) {
    if (n >= size) return `${(n / size).toFixed(2)} ${unit}FLOPs`;
  }
  return n ? `${n} FLOPs` : "—";
}

function BudgetPanel({ report }: { report: BudgetReport | null }) {
  const { meta, setMeta } = useStore();
  const [devices, setDevices] = useState<DeviceProfile[]>([]);
  useEffect(() => { api.devices().then((r) => setDevices(r.devices)).catch(() => undefined); }, []);
  const budget = { device: "", max_train_memory_gb: 0, max_latency_ms: 0, max_params_m: 0,
                   max_model_mb: 0, ...((meta.budget as Partial<Budget>) ?? {}) };
  const set = (patch: Partial<Budget>) => setMeta((m) => ({ ...m, budget: { ...budget, ...patch } }));
  const memory = report?.checks.find((c) => c.kind === "train_memory");
  return (
    <fieldset className="budget" data-testid="budget-panel">
      <legend>Budget</legend>
      <label>Device <select value={budget.device} aria-label="Target device"
                            onChange={(e) => set({ device: e.target.value })}>
        <option value="">No target device</option>
        {devices.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
      </select></label>
      {LIMITS.map((l) => (
        <label key={l.key}>{l.label} <input type="number" min={0} step="any" style={{ width: 80 }}
          value={Number(budget[l.key]) || ""} placeholder="no limit" aria-label={l.label}
          onChange={(e) => set({ [l.key]: Number(e.target.value) || 0 })} /> {l.unit}</label>
      ))}
      {memory && (
        <div className={`budget-bar${memory.over ? " over" : ""}`} data-testid="memory-bar"
             title={`${memory.used.toFixed(2)} of ${memory.limit} GB`}>
          <span style={{ width: `${Math.min(100, (100 * memory.used) / memory.limit)}%` }} />
          <em>{memory.used.toFixed(2)} / {memory.limit} GB training memory</em>
        </div>
      )}
      {report?.checks.filter((c) => c.over && c.kind !== "train_memory").map((c) => (
        <p key={c.kind} className="error-text">{c.kind.replace("_", " ")} {c.used.toFixed(2)} {c.unit} is
          over the {c.limit} {c.unit} budget</p>
      ))}
    </fieldset>
  );
}

function SummaryTab() {
  const { project, validation } = useStore();
  const [summary, setSummary] = useState<Summary | null>(null);
  const [report, setReport] = useState<BudgetReport | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!validation?.valid) {
      setSummary(null);
      setReport(null);
      setError("Fix the errors to see the model summary.");
      return;
    }
    api.summary(project()).then((s) => { setSummary(s); setError(""); })
      .catch((e) => { setSummary(null); setError(String(e.message ?? e)); });
    api.budget(project()).then(setReport).catch(() => setReport(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validation]);
  if (!summary) return <p className="muted">{error}</p>;
  const est = report?.estimate;
  return (
    <>
      <div className="tiles" data-testid="cost-tiles">
        <div><strong>{summary.total_params_display}</strong><span>parameters</span></div>
        <div><strong>{flops(summary.total_flops)}</strong><span>forward / sample</span></div>
        <div><strong>{est ? `${est.train_memory_gb.toFixed(2)} GB` : "—"}</strong>
          <span>training memory{est ? ` · batch ${est.batch_size}` : ""}</span></div>
        <div><strong>{est?.latency_ms != null ? `${est.latency_ms.toFixed(2)} ms` : "—"}</strong>
          <span>latency{est?.device ? "" : " (pick a device)"}</span></div>
      </div>
      <BudgetPanel report={report} />
      <table>
        <thead><tr><th>Layer</th><th>Output</th><th className="num">Params</th>
          <th className="num">FLOPs</th></tr></thead>
        <tbody>
          {summary.layers.map((l, i) => (
            <tr key={i}><td>{l.name}</td><td>{shapeLabel(l.output_shape)}</td>
              <td className="num">{l.params.toLocaleString()}</td>
              <td className="num">{flops(l.flops)}</td></tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function CodeTab() {
  const { project, validation, name } = useStore();
  const [targets, setTargets] = useState<{ id: string; label: string }[]>([]);
  const [target, setTarget] = useState("pytorch_model");
  const [code, setCode] = useState("");
  useEffect(() => { api.targets().then((r) => setTargets(r.targets)).catch(() => undefined); }, []);
  useEffect(() => {
    api.generate(project(), target).then((r) => setCode(r.code))
      .catch((e) => setCode(`# ${e.message ?? e}`));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validation, target]);
  const download = () => {
    const blob = new Blob([code], { type: "text/x-python" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name || "model"}_${target}.py`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return (
    <>
      <div className="toolbar" style={{ marginBottom: 8 }}>
        <select value={target} onChange={(e) => setTarget(e.target.value)} aria-label="Code target">
          {targets.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
        </select>
        <button onClick={download} disabled={!code}>Download</button>
        <button onClick={() => navigator.clipboard.writeText(code)} disabled={!code}>Copy</button>
      </div>
      <CodeView code={code} />
    </>
  );
}

export function Inspector() {
  const [tab, setTab] = useState("properties");
  return (
    <aside className="panel inspector">
      <div className="tabs">
        {["properties", "summary", "code"].map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      <div className="tab-body">
        {tab === "properties" && <Properties />}
        {tab === "summary" && <SummaryTab />}
        {tab === "code" && <CodeTab />}
      </div>
    </aside>
  );
}
