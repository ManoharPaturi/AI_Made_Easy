// The tools you already use (3.2): experiment tracking for this project, datasets from
// Kaggle and the Hugging Face Hub, and the run history as a table for notebooks.
import { useEffect, useState } from "react";

import { api } from "../api";
import { useStore } from "../state";
import type { FetchedDataset, TrackingStatus } from "../types";
import { Modal } from "./Modal";

export function ToolsModal({ onClose, onUseData }: {
  onClose: () => void; onUseData: (data: FetchedDataset) => void;
}) {
  const { meta, setMeta, notify } = useStore();
  const [status, setStatus] = useState<TrackingStatus | null>(null);
  const [source, setSource] = useState("");
  const [busy, setBusy] = useState(false);
  const [fetched, setFetched] = useState<FetchedDataset | null>(null);
  const chosen: string[] = Array.isArray(meta.tracking) ? (meta.tracking as string[])
    : String(meta.tracking ?? "").split(",").map((t) => t.trim()).filter(Boolean);
  useEffect(() => {
    api.tracking().then(setStatus).catch((e) => notify(String(e.message), true));
  }, [notify]);
  const toggle = (id: string, on: boolean) => {
    const next = on ? [...new Set([...chosen, id])] : chosen.filter((t) => t !== id);
    setMeta((m) => {
      const copy = { ...m };
      if (next.length) copy.tracking = next; else delete copy.tracking;
      return copy;
    });
  };
  const download = async () => {
    if (!source.trim()) return;
    setBusy(true);
    setFetched(null);
    try {
      setFetched(await api.fetchDataset(source.trim()));
    } catch (e) {
      notify(String((e as Error).message), true);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="Tools: tracking, datasets, run history" onClose={onClose} wide>
      <section data-testid="tools-tracking">
        <h3>Experiment tracking</h3>
        <p className="dim">This project's runs mirror as they train, on every backend:
          parameters, per-epoch metrics, the design and the run's files. Sign in with the
          tool itself (<code>wandb login</code>, <code>MLFLOW_TRACKING_URI</code>); AI Made
          Easy never stores a token.</p>
        {status?.trackers.map((t) => (
          <label key={t.id} className="check">
            <input type="checkbox" data-testid={`track-${t.id}`} checked={chosen.includes(t.id)}
                   disabled={!t.installed && !chosen.includes(t.id)}
                   onChange={(e) => toggle(t.id, e.target.checked)} />
            {t.label}{!t.installed && <span className="muted"> (not installed: pip
              install 'ai-made-easy[tracking]')</span>}
          </label>))}
        {status && <p className="muted">MLflow store: <code>{status.mlflow_uri}</code>
          {status.env && <> · every run also uses <code>AIME_TRACKING={status.env}</code></>}</p>}
      </section>
      <section data-testid="tools-datasets">
        <h3>Datasets from Kaggle and the Hub</h3>
        <p className="dim"><code>kaggle:owner/name</code>, <code>kaggle-competition:name</code>{" "}
          or <code>hf:owner/repo</code>: downloaded once, then detected like a local folder.
          Kaggle uses the server's <code>~/.kaggle/kaggle.json</code>.</p>
        <div className="toolbar">
          <input value={source} placeholder="kaggle:owner/name" style={{ flex: 1 }}
                 data-testid="dataset-source" onChange={(e) => setSource(e.target.value)}
                 onKeyDown={(e) => e.key === "Enter" && download()} />
          <button onClick={download} disabled={busy || !source.trim()}
                  data-testid="dataset-fetch">{busy ? "Downloading…" : "Download"}</button>
        </div>
        {fetched && (
          <div className="card" data-testid="dataset-result">
            <code>{fetched.path}</code>
            <div className="dim">{fetched.facts.kind}: {fetched.facts.rows.toLocaleString()} rows
              {fetched.facts.tasks.length ? ` · fits ${fetched.facts.tasks.join(", ")}` : ""}</div>
            <button className="primary" onClick={() => onUseData(fetched)}
                    data-testid="dataset-use">
              New project from it…</button>
          </div>)}
      </section>
      <section data-testid="tools-history">
        <h3>Run history for notebooks</h3>
        <p className="dim">Every run with its parameters and metrics, or one row per epoch.</p>
        <div className="toolbar">
          <a href={api.exportRunsUrl("csv")} download><button>Runs (CSV)</button></a>
          <a href={api.exportRunsUrl("parquet")} download><button>Runs (Parquet)</button></a>
          <a href={api.exportRunsUrl("csv", true)} download><button>Epochs (CSV)</button></a>
        </div>
      </section>
    </Modal>
  );
}
