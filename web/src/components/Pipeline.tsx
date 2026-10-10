// Pipeline stage timeline and the panel that runs a pipeline design.
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import { useStore } from "../state";
import type { PipelineRecord } from "../types";
import { duration, metric, StatusChip } from "./Modal";

const RUNNING = ["created", "running"];

function summary(metrics: Record<string, number>): string {
  return Object.entries(metrics).filter(([k]) => !k.endsWith("_std") && !k.startsWith("member"))
    .slice(0, 4).map(([k, v]) => `${k} ${metric(v)}`).join(" · ");
}

export function StageTimeline({ record }: { record: PipelineRecord }) {
  return (
    <table data-testid="stage-timeline">
      <thead><tr><th>Stage</th><th>State</th><th>Result</th><th>Time</th><th>Run</th></tr></thead>
      <tbody>
        {record.order.map((id) => {
          const s = record.stages[id];
          const time = s.started_at && s.finished_at ? duration(s.finished_at - s.started_at) : "";
          return (
            <tr key={id} data-testid={`stage-${id}`}>
              <td><b>{s.label}</b> <span className="muted">{id}</span>
                {s.inputs.length > 0 && <span className="muted"> ← {s.inputs.join(", ")}</span>}</td>
              <td><StatusChip status={s.state === "cached" ? "finished" : s.state} />
                {s.state === "cached" && <span className="muted"> reused</span>}</td>
              <td className="dim">{summary(s.metrics)}{s.message && (
                <div className={s.state === "failed" ? "sev-error" : "muted"}>{s.message}</div>)}</td>
              <td className="num">{time}</td>
              <td className="muted">{s.run_id}</td>
            </tr>);
        })}
      </tbody>
    </table>
  );
}

export function usePipeline(initial: string | null = null) {
  const [id, setId] = useState<string | null>(initial);
  const [record, setRecord] = useState<PipelineRecord | null>(null);
  useEffect(() => {
    if (!id) return;
    let alive = true;
    const tick = () => api.pipeline(id).then((r) => {
      if (!alive) return;
      setRecord(r);
      if (RUNNING.includes(r.state)) window.setTimeout(tick, 1000);
    }).catch(() => undefined);
    tick();
    return () => { alive = false; };
  }, [id]);
  return { id, setId, record };
}

/** The Training tab for pipeline designs. */
export function PipelinePanel() {
  const { project, name, validation, notify } = useStore();
  const { setId, record } = usePipeline();
  const running = !!record && RUNNING.includes(record.state);
  const act = useCallback(async (fn: () => Promise<{ pipeline_id: string }>, what: string) => {
    try {
      const r = await fn();
      setId(r.pipeline_id);
      notify(`${what} — pipeline ${r.pipeline_id}`);
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  }, [setId, notify]);
  return (
    <div data-testid="pipeline-panel">
      <div className="toolbar" style={{ marginBottom: 6 }}>
        <button className="primary" data-testid="run-pipeline" disabled={running || !validation?.valid}
                onClick={() => act(() => api.startPipeline(project(), name), "Pipeline started")}>
          ▶ Run pipeline</button>
        <button disabled={!running} onClick={() => record && api.stopPipeline(record.pipeline_id)}>Stop</button>
        <button disabled={!record || running || record.state === "finished"}
                onClick={() => record && act(() => api.resumePipeline(record.pipeline_id), "Resumed")}>
          Resume</button>
        <span className="dim">{record ? `${record.pipeline_id} · ${record.state}` : "Stages run in order; "
          + "finished, unchanged stages from earlier runs are reused."}</span>
      </div>
      {record && <StageTimeline record={record} />}
    </div>
  );
}

/** Experiments ▸ Pipelines. */
export function PipelinesList() {
  const [rows, setRows] = useState<PipelineRecord[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const refresh = useCallback(() => {
    api.pipelines().then((r) => setRows(r.pipelines)).catch(() => undefined);
  }, []);
  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 4000);
    return () => window.clearInterval(timer);
  }, [refresh]);
  const current = rows.find((r) => r.pipeline_id === open);
  return (
    <>
      <table>
        <thead><tr><th>Pipeline</th><th>Status</th><th className="num">Stages</th><th>Started</th></tr></thead>
        <tbody>
          {rows.map((r) => {
            const done = r.order.filter((id) => ["finished", "cached"].includes(r.stages[id].state)).length;
            return (
              <tr key={r.pipeline_id} className={open === r.pipeline_id ? "selected" : ""}
                  onClick={() => setOpen(r.pipeline_id)} style={{ cursor: "pointer" }}>
                <td>{r.name} <span className="muted">{r.pipeline_id}</span>
                  {r.resumed_from && <span className="muted"> (resumes {r.resumed_from})</span>}</td>
                <td><StatusChip status={r.state} /></td>
                <td className="num">{done}/{r.order.length}</td>
                <td className="muted">{new Date(r.created_at * 1000).toLocaleString()}</td>
              </tr>);
          })}
        </tbody>
      </table>
      {!rows.length && <div className="empty">No pipelines yet: open a pipeline example and run it.</div>}
      {current && <><h3>Stages of {current.name}</h3><StageTimeline record={current} /></>}
    </>
  );
}
