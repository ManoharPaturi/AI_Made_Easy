import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import { LineChart, type Series } from "../components/LineChart";
import { duration, metric, Modal, StatusChip } from "../components/Modal";
import { useStore } from "../state";
import type { RunRecord, RunSamples, SweepParam, SweepRecord } from "../types";

function DeployModal({ run, onClose }: { run: RunRecord; onClose: () => void }) {
  const { notify } = useStore();
  const [formats, setFormats] = useState<{ id: string; label: string }[]>([]);
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [name, setName] = useState(run.name);
  useEffect(() => {
    api.formats(run.run_id).then((r) => setFormats(r.formats)).catch(() => undefined);
  }, [run.run_id]);
  const register = async () => {
    try {
      const v = await api.registerModel(run.run_id, name);
      notify(`Registered ${v.name} v${v.version}`);
      onClose();
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  return (
    <Modal title={`Deploy run ${run.run_id}`} onClose={onClose} actions={
      <>
        <button onClick={register}>Register as model</button>
        <a href={api.deployUrl(run.run_id, [...chosen])} download>
          <button className="primary">Download serving package</button></a>
      </>}>
      <p className="dim">The package contains a FastAPI server (<code>/predict</code>,
        <code> /health</code>, <code>/metadata</code>), the trained model with its fitted
        preprocessing, a Dockerfile and a README.</p>
      <div className="form">
        <div className="field"><label htmlFor="model-name">Model name</label>
          <input id="model-name" value={name} onChange={(e) => setName(e.target.value)} /></div>
        <div className="field"><label>Extra export formats</label>
          {formats.map((f) => (
            <label key={f.id} style={{ display: "flex", gap: 8, alignItems: "center", color: "var(--text)" }}>
              <input type="checkbox" checked={chosen.has(f.id)} onChange={(e) => setChosen((s) => {
                const next = new Set(s);
                if (e.target.checked) next.add(f.id); else next.delete(f.id);
                return next;
              })} />{f.label}</label>
          ))}
        </div>
      </div>
    </Modal>
  );
}

function Runs({ scope }: { scope: string }) {
  const { load, notify } = useStore();
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [curves, setCurves] = useState<Series[]>([]);
  const [compare, setCompare] = useState<Record<string, unknown> | null>(null);
  const [deploying, setDeploying] = useState<RunRecord | null>(null);

  const refresh = useCallback(() => {
    api.runs(scope || undefined).then((r) => setRuns(r.runs)).catch((e) => notify(String(e.message), true));
  }, [scope, notify]);
  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 4000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const ids = [...selected];
  useEffect(() => {
    if (!ids.length) { setCurves([]); setCompare(null); return; }
    Promise.all(ids.map((id) => api.metrics(id).then((m) => [id, m.epochs] as const)))
      .then((rows) => setCurves(rows.flatMap(([id, epochs]) => {
        const keys = [...new Set(epochs.flatMap((e) => Object.keys(e.metrics)))]
          .filter((k) => k === "val_loss" || k === "loss" || k === "train_loss");
        return keys.slice(0, 1).map((k) => ({
          name: `${id} ${k}`, points: epochs.map((e) => [e.epoch, e.metrics[k]] as [number, number]),
        }));
      })))
      .catch(() => setCurves([]));
    if (ids.length > 1) api.compare(ids).then(setCompare).catch(() => setCompare(null));
    else setCompare(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  const toggle = (id: string) => setSelected((s) => {
    const next = new Set(s);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const restore = async (id: string) => {
    const run = await api.run(id);
    if (run.graph) { load(run.graph); notify(`Restored the design of run ${id}`); }
  };
  const remove = async () => {
    if (!confirm(`Delete ${ids.length} run(s) and their files?`)) return;
    await Promise.all(ids.map((id) => api.deleteRun(id)));
    setSelected(new Set());
    refresh();
  };
  const metricKeys = [...new Set(runs.flatMap((r) => Object.keys(r.final_metrics)))].slice(0, 4);
  return (
    <>
      <div className="toolbar" style={{ marginBottom: 8 }}>
        <button onClick={refresh}>Refresh</button>
        <button disabled={ids.length !== 1} onClick={() => restore(ids[0])}>Restore design</button>
        <button disabled={ids.length !== 1 || runs.find((r) => r.run_id === ids[0])?.status !== "finished"}
                onClick={() => setDeploying(runs.find((r) => r.run_id === ids[0]) ?? null)}>Deploy…</button>
        <button className="danger" disabled={!ids.length} onClick={remove}>Delete</button>
        <span className="muted">{runs.length} run(s){ids.length > 1 && ` · comparing ${ids.length}`}</span>
      </div>
      <table data-testid="runs">
        <thead><tr><th /><th>Run</th><th>Name</th><th>Status</th><th>Framework</th>
          {metricKeys.map((k) => <th key={k} className="num">{k}</th>)}
          <th className="num">Epochs</th><th className="num">Duration</th><th>Tags</th></tr></thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run_id} className={selected.has(r.run_id) ? "selected" : ""}
                onClick={() => toggle(r.run_id)} style={{ cursor: "pointer" }}>
              <td><input type="checkbox" readOnly checked={selected.has(r.run_id)} /></td>
              <td className="dim">{r.run_id}</td><td>{r.name}</td>
              <td><StatusChip status={r.status} /></td><td>{r.framework}</td>
              {metricKeys.map((k) => <td key={k} className="num">{metric(r.final_metrics[k])}</td>)}
              <td className="num">{r.epochs_done}</td><td className="num">{duration(r.duration)}</td>
              <td>{r.tags.map((t) => <span key={t} className="chip">{t}</span>)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {!runs.length && <div className="empty">No runs yet — train a design to record one.</div>}
      {curves.length > 0 && <><h3>Loss curves</h3><LineChart series={curves} /></>}
      {compare && <CompareTable data={compare} />}
      {ids.length === 1 && <RunSamplesView runId={ids[0]} />}
      {deploying && <DeployModal run={deploying} onClose={() => setDeploying(null)} />}
    </>
  );
}

function RunSamplesView({ runId }: { runId: string }) {
  const [data, setData] = useState<RunSamples | null>(null);
  useEffect(() => {
    setData(null);
    api.runSamples(runId).then(setData).catch(() => setData(null));
  }, [runId]);
  const texts = data?.texts ?? [];
  if (!data || (!data.samples.length && !texts.length)) return null;
  const offset = data.per_class.length === data.classes.length + 1 ? 1 : 0;
  const caption = data.per_class_metric === "AP"
    ? "Green: ground truth · red: prediction with its score."
    : data.per_class_metric === "IoU" ? "Each sample: true mask (left) and prediction (right)."
      : "What the trained model produced on held-out data or from noise.";
  return (
    <div data-testid="run-samples">
      <h3>{data.per_class_metric ? "Test predictions" : "Samples"}</h3>
      <p className="muted">{caption}</p>
      {data.per_class.length > 0 && (
        <table style={{ maxWidth: 420 }}>
          <thead><tr><th>Class</th><th className="num">{data.per_class_metric}</th></tr></thead>
          <tbody>{data.per_class.map((v, i) => (
            <tr key={i}><td>{i < offset ? "background" : data.classes[i - offset] ?? i}</td>
              <td className="num">{v == null ? "—" : v.toFixed(3)}</td></tr>))}</tbody>
        </table>)}
      <div className="sample-grid">
        {data.samples.map((s) => <img key={s.name} src={s.data_url} alt={s.name} title={s.name} />)}
      </div>
      {texts.map((t) => (
        <pre key={t.name} className="code" style={{ whiteSpace: "pre-wrap", maxHeight: 320,
                                                    overflow: "auto" }}>{t.text}</pre>))}
    </div>
  );
}

function CompareTable({ data }: { data: Record<string, unknown> }) {
  const runs = data.runs as string[];
  const rows: [string, string, unknown[]][] = [];
  for (const [k, v] of Object.entries((data.final_metrics ?? {}) as Record<string, unknown[]>)) rows.push(["metric", k, v]);
  for (const [k, v] of Object.entries((data.params ?? {}) as Record<string, unknown[]>)) rows.push(["parameter", k, v]);
  rows.push(["run", "data", (data.data as string[]).map((d) => d || "—")]);
  return (
    <>
      <h3>Comparison {data.same_data === false && <span className="sev-warning">— trained on different data</span>}</h3>
      <table>
        <thead><tr><th>Section</th><th>Name</th>{runs.map((r) => <th key={r}>{r}</th>)}</tr></thead>
        <tbody>
          {rows.map(([section, name, values]) => (
            <tr key={section + name}><td className="muted">{section}</td><td>{name}</td>
              {values.map((v, i) => <td key={i}>{typeof v === "number" ? metric(v)
                : v == null ? "—" : typeof v === "string" ? v : JSON.stringify(v)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function SweepModal({ onClose, onStarted }: { onClose: () => void; onStarted: () => void }) {
  const { project, name, notify } = useStore();
  const [params, setParams] = useState<SweepParam[]>([]);
  const [chosen, setChosen] = useState<Record<string, SweepParam>>({});
  const [metricName, setMetricName] = useState("val_loss");
  const [strategy, setStrategy] = useState("random");
  const [trials, setTrials] = useState(6);
  const [skipBudget, setSkipBudget] = useState(true);
  useEffect(() => {
    api.sweepParams(project()).then((r) => setParams(r.params)).catch((e) => notify(String(e.message), true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const start = async () => {
    const dimensions = Object.values(chosen).map((p) => (p.kind === "choice"
      ? { node: p.node, param: p.param, kind: "choice", values: p.values }
      : { node: p.node, param: p.param, kind: p.kind, low: p.low, high: p.high, log: p.log }));
    try {
      await api.startSweep(project(), { dimensions, metric: metricName, strategy,
                                        max_trials: trials, skip_over_budget: skipBudget }, name);
      notify("Sweep started");
      onStarted();
      onClose();
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const update = (key: string, patch: Partial<SweepParam>) =>
    setChosen((c) => ({ ...c, [key]: { ...c[key], ...patch } }));
  return (
    <Modal title="New hyperparameter sweep" onClose={onClose} actions={
      <button className="primary" disabled={!Object.keys(chosen).length} onClick={start}>Start sweep</button>}>
      <div className="toolbar" style={{ marginBottom: 10 }}>
        <label>Objective <input value={metricName} onChange={(e) => setMetricName(e.target.value)}
                                style={{ width: 120 }} /></label>
        <label>Strategy <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
          <option value="random">Random</option><option value="grid">Grid</option>
          <option value="tpe">Bayesian (Optuna TPE)</option></select></label>
        <label>Trials <input type="number" min={1} max={200} value={trials}
                             onChange={(e) => setTrials(Number(e.target.value))} style={{ width: 70 }} /></label>
        <label title="Trials that break the project's resource budget are recorded but not trained">
          <input type="checkbox" checked={skipBudget} onChange={(e) => setSkipBudget(e.target.checked)} />
          Skip over budget</label>
      </div>
      <table>
        <thead><tr><th /><th>Block · parameter</th><th>Range</th></tr></thead>
        <tbody>
          {params.filter((p) => p.kind !== "choice" || (p.values?.length ?? 0) > 1).map((p) => {
            const on = p.key in chosen;
            const c = chosen[p.key] ?? p;
            return (
              <tr key={p.key}>
                <td><input type="checkbox" checked={on} onChange={(e) => setChosen((s) => {
                  const next = { ...s };
                  if (e.target.checked) next[p.key] = p; else delete next[p.key];
                  return next;
                })} /></td>
                <td>{p.block} · {p.param} <span className="muted">({String(p.current)})</span></td>
                <td>{p.kind === "choice" ? (p.values ?? []).map(String).join(" / ") : (
                  <span className="toolbar">
                    <input type="number" disabled={!on} value={c.low} style={{ width: 90 }}
                           onChange={(e) => update(p.key, { low: Number(e.target.value) })} />–
                    <input type="number" disabled={!on} value={c.high} style={{ width: 90 }}
                           onChange={(e) => update(p.key, { high: Number(e.target.value) })} />
                    {p.kind === "float" && <label><input type="checkbox" disabled={!on} checked={!!c.log}
                      onChange={(e) => update(p.key, { log: e.target.checked })} /> log</label>}
                  </span>)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Modal>
  );
}

function Sweeps() {
  const { load, notify } = useStore();
  const [sweeps, setSweeps] = useState<SweepRecord[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const refresh = useCallback(() => {
    api.sweeps().then((r) => setSweeps(r.sweeps)).catch(() => undefined);
  }, []);
  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 4000);
    return () => window.clearInterval(timer);
  }, [refresh]);
  const apply = async (id: string) => {
    try {
      load((await api.bestGraph(id)).graph);
      notify(sweeps.find((s) => s.sweep_id === id)?.spec.automl
        ? "Opened the best AutoML design" : "Applied the best trial's parameters to the design");
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const sweep = sweeps.find((s) => s.sweep_id === open);
  return (
    <>
      <div className="toolbar" style={{ marginBottom: 8 }}>
        <button className="primary" onClick={() => setCreating(true)}>New sweep…</button>
        <button onClick={refresh}>Refresh</button>
      </div>
      <table>
        <thead><tr><th>Sweep</th><th>Status</th><th>Objective</th><th>Strategy</th>
          <th className="num">Trials</th><th className="num">Best</th><th /></tr></thead>
        <tbody>
          {sweeps.map((s) => (
            <tr key={s.sweep_id} className={open === s.sweep_id ? "selected" : ""}
                onClick={() => setOpen(s.sweep_id)} style={{ cursor: "pointer" }}>
              <td>{s.name} <span className="muted">{s.sweep_id}</span></td>
              <td><StatusChip status={s.state} /></td>
              <td>{s.spec.metric} ({s.spec.direction || "auto"})</td>
              <td>{s.spec.automl ? <span className="chip run" title={(s.spec.candidates ?? []).join(", ")}>
                AutoML · {s.spec.candidates?.length ?? 0} recipes</span> : s.spec.strategy}</td>
              <td className="num">{s.trials.length}/{s.spec.max_trials}</td>
              <td className="num">{metric(s.best?.score)}</td>
              <td>{s.state === "running"
                ? <button onClick={(e) => { e.stopPropagation(); api.stopSweep(s.sweep_id).then(refresh); }}>Stop</button>
                : s.best && <button onClick={(e) => { e.stopPropagation(); apply(s.sweep_id); }}>
                  {s.spec.automl ? "Open best design" : "Apply best"}</button>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {!sweeps.length && <div className="empty">No sweeps yet.</div>}
      {sweep && (
        <>
          <h3>Trials of {sweep.name}</h3>
          <table>
            <thead><tr><th>#</th><th>State</th><th>Values</th><th className="num">Score</th><th>Run</th></tr></thead>
            <tbody>
              {sweep.trials.map((t) => (
                <tr key={t.number}><td>{t.number}</td><td><StatusChip status={t.state} /></td>
                  <td className="dim">{Object.entries(t.values).map(([k, v]) => `${k}=${metric(v) === "—" ? String(v) : metric(v)}`).join(", ")}</td>
                  <td className="num">{metric(t.score)}</td><td className="muted">{t.run_id || t.message}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {creating && <SweepModal onClose={() => setCreating(false)} onStarted={refresh} />}
    </>
  );
}

export function ExperimentsPage({ initialTab = "runs" }: { initialTab?: string }) {
  const { name } = useStore();
  const [tab, setTab] = useState(initialTab);
  const [scoped, setScoped] = useState(true);
  return (
    <div className="page" data-testid="experiments-page">
      <div className="toolbar" style={{ marginBottom: 10 }}>
        <h2 style={{ margin: 0 }}>Experiments</h2>
        <div className="tabs" style={{ border: "none", padding: 0 }}>
          {["runs", "sweeps"].map((t) => (
            <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
              {t[0].toUpperCase() + t.slice(1)}</button>
          ))}
        </div>
        {tab === "runs" && (
          <label className="dim"><input type="checkbox" checked={scoped}
                                        onChange={(e) => setScoped(e.target.checked)} /> only “{name}”</label>
        )}
      </div>
      {tab === "runs" ? <Runs scope={scoped ? name : ""} /> : <Sweeps />}
    </div>
  );
}
