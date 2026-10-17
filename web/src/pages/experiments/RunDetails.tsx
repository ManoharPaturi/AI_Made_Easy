// A run's details on the Experiments page: deploy, feature importance, design diff, samples,
// and the comparison table.
import { useEffect, useState } from "react";

import { api } from "../../api";
import { metric, Modal } from "../../components/Modal";
import { useStore } from "../../state";
import type { DesignDiff, ExportFormat, FeatureImportance, RunRecord, RunSamples } from "../../types";


export function DeployModal({ run, onClose }: { run: RunRecord; onClose: () => void }) {
  const { notify } = useStore();
  const [formats, setFormats] = useState<ExportFormat[]>([]);
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [name, setName] = useState(run.name);
  useEffect(() => {
    api.formats(run.run_id).then((r) => {
      setFormats(r.formats);
      setChosen(new Set(r.formats.filter((f) => f.suggested).map((f) => f.id)));
    }).catch(() => undefined);
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
          {formats.map((f) => {
            const blocked = f.available === false;
            return (
              <label key={f.id} className={`export-format${blocked ? " blocked" : ""}`}
                title={blocked ? f.reason : undefined} data-format={f.id}>
                <input type="checkbox" disabled={blocked} checked={!blocked && chosen.has(f.id)}
                  onChange={(e) => setChosen((s) => {
                    const next = new Set(s);
                    if (e.target.checked) next.add(f.id); else next.delete(f.id);
                    return next;
                  })} />
                <span>{f.label}</span>
                {f.suggested && <span className="chip">suggested for {f.device}</span>}
                {blocked && <span className="dim export-reason">{f.reason}</span>}
              </label>
            );
          })}
          <p className="dim">Every export is checked against the trained model; the package's
            <code> export_report.json</code> has the comparison.</p>
        </div>
      </div>
    </Modal>
  );
}

export function RunExplainView({ runId }: { runId: string }) {
  const [data, setData] = useState<FeatureImportance | null>(null);
  useEffect(() => {
    setData(null);
    api.runExplain(runId).then(setData).catch(() => setData(null));
  }, [runId]);
  if (!data?.features.length) return null;
  const top = data.features.slice(0, 15);
  const max = Math.max(...top.map((f) => Math.abs(f.importance)), 1e-12);
  return (
    <div data-testid="run-explain">
      {data.method === "integrated_gradients" ? <>
        <h3>Feature attribution</h3>
        <p className="muted">Integrated gradients: how much each feature moved the prediction,
          on average over {data.rows ?? "?"} test rows (from the {String(data.baseline ?? "mean")} row).</p>
      </> : <>
        <h3>Feature importance</h3>
        <p className="muted">How much the {data.metric ?? "test loss"} rises when each feature is
          shuffled ({data.rows ?? "?"} test rows): the features the model relies on most.</p>
      </>}
      <table style={{ maxWidth: 560 }}>
        <tbody>{top.map((f) => (
          <tr key={f.name}>
            <td style={{ whiteSpace: "nowrap" }}>{f.name}</td>
            <td style={{ width: "60%" }}>
              <div className="importance-bar" style={{ width: `${Math.max(0, f.importance) / max * 100}%` }} />
            </td>
            <td className="num">{f.importance.toFixed(4)}<span className="muted"> ± {f.std.toFixed(4)}</span></td>
          </tr>))}</tbody>
      </table>
    </div>
  );
}

const show = (v: unknown) => (typeof v === "string" ? v : JSON.stringify(v));

/** The canvas compared with the design a run trained. */
export function DiffModal({ runId, onClose }: { runId: string; onClose: () => void }) {
  const { project, notify } = useStore();
  const [d, setD] = useState<DesignDiff | null>(null);
  useEffect(() => {
    api.runDiff(runId, project()).then(setD).catch((e) => notify(String(e.message), true));
  }, [runId, project, notify]);
  return (
    <Modal title={`Run ${runId} vs the canvas`} onClose={onClose} wide>
      {!d ? <p className="muted">Comparing…</p> : (
        <div data-testid="design-diff">
          <p><b>{d.summary}</b></p>
          {d.changed.length > 0 && (
            <table><thead><tr><th>Block</th><th>Parameter</th><th>In the run</th><th>Now</th></tr></thead>
              <tbody>{d.changed.map((c) => (
                <tr key={`${c.id}.${c.param}`}><td>{c.id}</td><td>{c.param}</td>
                  <td className="dim">{show(c.from)}</td><td>{show(c.to)}</td></tr>))}</tbody></table>)}
          {[["Added", d.added.map((x) => `${x.id} (${x.type})`)],
            ["Removed", d.removed.map((x) => `${x.id} (${x.type})`)],
            ["Replaced", d.retyped.map((x) => `${x.id}: ${x.from} → ${x.to}`)],
            ["New wires", d.wired.map((w) => `${w.from} → ${w.to}`)],
            ["Removed wires", d.unwired.map((w) => `${w.from} → ${w.to}`)]]
            .filter(([, items]) => (items as string[]).length)
            .map(([label, items]) => (
              <div key={label as string}><h3>{label as string}</h3>
                <ul>{(items as string[]).map((i) => <li key={i}>{i}</li>)}</ul></div>))}
        </div>)}
    </Modal>
  );
}

export function RunSamplesView({ runId }: { runId: string }) {
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

export function CompareTable({ data }: { data: Record<string, unknown> }) {
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
