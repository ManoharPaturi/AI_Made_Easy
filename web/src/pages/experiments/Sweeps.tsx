// Sweeps and AutoML on the Experiments page: start a sweep, trials, the Pareto front.
import { useCallback, useEffect, useState } from "react";

import { api } from "../../api";
import { metric, Modal, StatusChip } from "../../components/Modal";
import { useStore } from "../../state";
import { RunOnSelect, useRunOn } from "../../components/Backends";
import type { SweepParam, SweepRecord, SweepTrial } from "../../types";


export function SweepModal({ onClose, onStarted }: { onClose: () => void; onStarted: () => void }) {
  const { project, name, notify } = useStore();
  const [params, setParams] = useState<SweepParam[]>([]);
  const [chosen, setChosen] = useState<Record<string, SweepParam>>({});
  const [metricName, setMetricName] = useState("val_loss");
  const [strategy, setStrategy] = useState("random");
  const [trials, setTrials] = useState(6);
  const [skipBudget, setSkipBudget] = useState(true);
  const [on] = useRunOn();
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
                                        max_trials: trials, skip_over_budget: skipBudget,
                                        on: on === "local" ? "" : on }, name);
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
        <RunOnSelect testId="sweep-run-on" />
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

/** Trials no other trial beats on the metric and every cost (multi-objective AutoML). */
export function paretoFront(sweep: SweepRecord): Set<number> {
  const costs = sweep.spec.objectives ?? [];
  if (!costs.length) return new Set();
  const sign = sweep.spec.direction === "min" ? 1 : -1;
  const points = sweep.trials
    .filter((t) => t.score !== null && costs.every((c) => t.objectives?.[c] !== undefined))
    .map((t) => ({ n: t.number, v: [sign * (t.score as number), ...costs.map((c) => t.objectives![c])] }));
  const beats = (a: number[], b: number[]) =>
    a.every((x, i) => x <= b[i]) && a.some((x, i) => x < b[i]);
  return new Set(points.filter((p) => !points.some((q) => q !== p && beats(q.v, p.v))).map((p) => p.n));
}

export function ParetoChart({ sweep, front }: { sweep: SweepRecord; front: Set<number> }) {
  const cost = sweep.spec.objectives![0];
  const pts = sweep.trials.filter((t): t is SweepTrial & { score: number } =>
    t.score !== null && t.objectives?.[cost] !== undefined);
  if (!pts.length) return null;
  const W = 420, H = 220, P = 40;
  const xs = pts.map((t) => t.objectives![cost]), ys = pts.map((t) => t.score);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x: number) => P + ((x - x0) / (x1 - x0 || 1)) * (W - P - 10);
  const sy = (y: number) => H - P + 10 - ((y - y0) / (y1 - y0 || 1)) * (H - P - 10);
  const on = pts.filter((t) => front.has(t.number)).sort((a, b) => a.objectives![cost] - b.objectives![cost]);
  return (
    <svg width={W} height={H} data-testid="pareto-chart" style={{ display: "block", margin: "8px 0" }}>
      <line x1={P} y1={H - P + 10} x2={W - 10} y2={H - P + 10} stroke="var(--border)" />
      <line x1={P} y1={10} x2={P} y2={H - P + 10} stroke="var(--border)" />
      <text x={W / 2} y={H - 4} textAnchor="middle" fill="var(--text-dim)" fontSize="11">{cost} (lower is better)</text>
      <text x={12} y={H / 2} textAnchor="middle" fontSize="11" fill="var(--text-dim)"
            transform={`rotate(-90 12 ${H / 2})`}>{sweep.spec.metric}</text>
      <polyline fill="none" stroke="var(--accent)" strokeWidth={2}
                points={on.map((t) => `${sx(t.objectives![cost])},${sy(t.score)}`).join(" ")} />
      {pts.map((t) => (
        <circle key={t.number} cx={sx(t.objectives![cost])} cy={sy(t.score)} r={front.has(t.number) ? 5 : 3.5}
                fill={front.has(t.number) ? "var(--accent)" : "var(--text-muted)"}>
          <title>{`#${t.number} ${String(t.values.recipe ?? "")}: ${sweep.spec.metric} ${metric(t.score)}, ${cost} ${metric(t.objectives![cost])}`}</title>
        </circle>))}
    </svg>
  );
}

export function Sweeps() {
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
  const front = sweep ? paretoFront(sweep) : new Set<number>();
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
              <td>{s.spec.metric} ({s.spec.direction || "auto"})
                {(s.spec.objectives ?? []).map((o) => <span key={o} className="chip" title={`minimised on ${s.spec.device}`}>
                  + {o}</span>)}</td>
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
          {!!sweep.spec.objectives?.length && (
            <>
              <p className="dim">◆ Pareto front: no other trial is better on {sweep.spec.metric} and
                {" "}{sweep.spec.objectives.join(" and ")} (estimated on {sweep.spec.device}).</p>
              <ParetoChart sweep={sweep} front={front} />
            </>)}
          <table>
            <thead><tr><th>#</th><th>State</th><th>Values</th><th className="num">Score</th>
              {!!sweep.spec.objectives?.length && <th>Costs</th>}<th>Run</th></tr></thead>
            <tbody>
              {sweep.trials.map((t) => (
                <tr key={t.number} data-pareto={front.has(t.number) || undefined}>
                  <td>{t.number}{front.has(t.number) && <span title="on the Pareto front"> ◆</span>}</td>
                  <td><StatusChip status={t.state} /></td>
                  <td className="dim">{Object.entries(t.values).map(([k, v]) => `${k}=${metric(v) === "—" ? String(v) : metric(v)}`).join(", ")}</td>
                  <td className="num">{metric(t.score)}</td>
                  {!!sweep.spec.objectives?.length && <td className="dim">
                    {sweep.spec.objectives.map((o) => `${o}=${metric(t.objectives?.[o])}`).join(", ")}</td>}
                  <td className="muted">{t.run_id || t.message}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {creating && <SweepModal onClose={() => setCreating(false)} onStarted={refresh} />}
    </>
  );
}
