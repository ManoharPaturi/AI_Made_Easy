import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import { metric } from "../components/Modal";
import { useStore } from "../state";
import type { ModelVersion } from "../types";

const STAGES = ["none", "staging", "production", "archived"];

export function ModelsPage() {
  const { notify } = useStore();
  const [models, setModels] = useState<ModelVersion[]>([]);
  const refresh = useCallback(() => {
    api.models().then((r) => setModels(r.models)).catch((e) => notify(String(e.message), true));
  }, [notify]);
  useEffect(refresh, [refresh]);
  const stage = async (m: ModelVersion, value: string) => {
    try {
      await api.setStage(m.name, m.version, value);
      refresh();
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const remove = async (m: ModelVersion) => {
    if (!confirm(`Delete ${m.name} v${m.version}?`)) return;
    await api.deleteModel(m.name, m.version).catch((e) => notify(String(e.message), true));
    refresh();
  };
  return (
    <div className="page" data-testid="models-page">
      <h2>Model registry</h2>
      <p className="dim">Register finished runs from Experiments ▸ Deploy. Promote a version to
        staging or production and download its serving package (FastAPI + Dockerfile).</p>
      <table>
        <thead><tr><th>Model</th><th className="num">Version</th><th>Stage</th><th>Framework</th>
          <th>Metrics</th><th>Run</th><th>Registered</th><th /></tr></thead>
        <tbody>
          {models.map((m) => (
            <tr key={`${m.name}-${m.version}`}>
              <td><strong>{m.name}</strong>{m.description && <div className="muted">{m.description}</div>}</td>
              <td className="num">v{m.version}</td>
              <td><select value={m.stage} onChange={(e) => stage(m, e.target.value)}
                          aria-label={`Stage of ${m.name} v${m.version}`}>
                {STAGES.map((s) => <option key={s} value={s}>{s}</option>)}</select></td>
              <td>{m.framework}</td>
              <td className="dim">{Object.entries(m.metrics).slice(0, 3)
                .map(([k, v]) => `${k} ${metric(v)}`).join(" · ")}</td>
              <td className="muted">{m.run_id}</td>
              <td className="muted">{new Date(m.created_at * 1000).toLocaleString()}</td>
              <td className="toolbar">
                <a href={`/api/models/${encodeURIComponent(m.name)}/${m.version}/deploy`} download>
                  <button>Download package</button></a>
                <button className="danger" onClick={() => remove(m)}>Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!models.length && <div className="empty">No registered models yet.</div>}
    </div>
  );
}
