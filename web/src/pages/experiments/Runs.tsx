// The runs table on the Experiments page.
import { useCallback, useEffect, useState } from "react";

import { api } from "../../api";
import { LineChart, type Series } from "../../components/LineChart";
import { useDialogs } from "../../components/Dialogs";
import { duration, ListState, metric, StatusChip } from "../../components/Modal";
import { useStore } from "../../state";
import type { RunRecord } from "../../types";

import { CompareTable, DeployModal, DiffModal, RunExplainView, RunSamplesView } from "./RunDetails";
import { TryModal } from "./TryIt";

export function Runs({ scope }: { scope: string }) {
  const { load, notify } = useStore();
  const { confirm } = useDialogs();
  const [loaded, setRuns] = useState<RunRecord[] | null>(null);
  const runs = loaded ?? [];
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [curves, setCurves] = useState<Series[]>([]);
  const [compare, setCompare] = useState<Record<string, unknown> | null>(null);
  const [deploying, setDeploying] = useState<RunRecord | null>(null);
  const [trying, setTrying] = useState<RunRecord | null>(null);
  const [diffing, setDiffing] = useState("");

  const refresh = useCallback(() => {
    api.runs(scope || undefined).then((r) => { setRuns(r.runs); setError(""); })
      .catch((e) => setError(String(e.message)));
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
    if (!(await confirm(`Delete ${ids.length} run(s) and their files?`,
                        { title: "Delete runs", ok: "Delete", danger: true }))) return;
    await Promise.all(ids.map((id) => api.deleteRun(id)));
    setSelected(new Set());
    refresh();
  };
  const importZip = async (file: File | undefined) => {
    if (!file) return;
    try {
      const r = await api.importResults(file);
      notify(`Imported the results of run ${r.run_id} (${r.state})`);
      refresh();
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const one = ids.length === 1 ? runs.find((r) => r.run_id === ids[0]) : undefined;
  const follow = async () => {
    if (!one) return;
    try {
      await api.attachRun(one.run_id);
      notify(`Following run ${one.run_id} again`);
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const metricKeys = [...new Set(runs.flatMap((r) => Object.keys(r.final_metrics)))].slice(0, 4);
  return (
    <>
      <div className="toolbar" style={{ marginBottom: 8 }}>
        <button onClick={refresh}>Refresh</button>
        <button disabled={ids.length !== 1} onClick={() => restore(ids[0])}>Restore design</button>
        <button disabled={ids.length !== 1} onClick={() => setDiffing(ids[0])} data-testid="diff-run"
                title="What changed on the canvas since this run">Compare with canvas</button>
        <button disabled={ids.length !== 1 || runs.find((r) => r.run_id === ids[0])?.status !== "finished"}
                onClick={() => setDeploying(runs.find((r) => r.run_id === ids[0]) ?? null)}>Deploy…</button>
        <button disabled={ids.length !== 1 || runs.find((r) => r.run_id === ids[0])?.status !== "finished"}
                onClick={() => setTrying(runs.find((r) => r.run_id === ids[0]) ?? null)}
                title="Run the trained model on your own input" data-testid="try-run">Try it…</button>
        <button disabled={!one || !one.backend || one.status !== "running"} onClick={follow}
                title="Follow a remote run again after a restart or a lost connection">Follow again</button>
        <label className="button" title="The results zip a Colab hand-off notebook downloaded">
          Import results…<input type="file" accept=".zip" hidden data-testid="import-results"
                                onChange={(e) => importZip(e.target.files?.[0])} /></label>
        <button className="danger" disabled={!ids.length} onClick={remove}>Delete</button>
        <span className="muted">{runs.length} run(s){ids.length > 1 && ` · comparing ${ids.length}`}</span>
      </div>
      <table data-testid="runs">
        <thead><tr><th /><th>Run</th><th>Name</th><th>Status</th><th>Framework</th><th>Ran on</th>
          {metricKeys.map((k) => <th key={k} className="num">{k}</th>)}
          <th className="num">Epochs</th><th className="num">Duration</th><th>Tags</th></tr></thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run_id} className={selected.has(r.run_id) ? "selected" : ""}
                onClick={() => toggle(r.run_id)} style={{ cursor: "pointer" }}>
              <td><input type="checkbox" readOnly checked={selected.has(r.run_id)} /></td>
              <td className="dim">{r.run_id}</td><td>{r.name}</td>
              <td><StatusChip status={r.status} /></td><td>{r.framework}</td>
              <td className="dim">{r.backend || "here"}</td>
              {metricKeys.map((k) => <td key={k} className="num">{metric(r.final_metrics[k])}</td>)}
              <td className="num">{r.epochs_done}</td><td className="num">{duration(r.duration)}</td>
              <td>{r.tags.map((t) => <span key={t} className="chip">{t}</span>)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <ListState items={loaded} error={error} onRetry={refresh} what="runs"
                 empty="No runs yet — train a design to record one." />
      {curves.length > 0 && <><h3>Loss curves</h3><LineChart series={curves} /></>}
      {compare && <CompareTable data={compare} />}
      {ids.length === 1 && <RunExplainView runId={ids[0]} />}
      {ids.length === 1 && <RunSamplesView runId={ids[0]} />}
      {diffing && <DiffModal runId={diffing} onClose={() => setDiffing("")} />}
      {deploying && <DeployModal run={deploying} onClose={() => setDeploying(null)} />}
      {trying && <TryModal run={trying} onClose={() => setTrying(null)} />}
    </>
  );
}

/** Permutation importance a run saved (the Feature Importance block). */
