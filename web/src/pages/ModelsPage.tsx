import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import { useDialogs } from "../components/Dialogs";
import { ListState, Modal, metric } from "../components/Modal";
import { useStore } from "../state";
import type { ExportFormat, ModelVersion } from "../types";

const STAGES = ["none", "staging", "production", "archived"];

// Where a registered version can go (3.2). Publishing to the Hub is confirmed every time;
// MLflow and W&B use their own sign-in, the Hub uses `hf auth login` on the server.
function ShareModal({ model, onClose, onChanged }: {
  model: ModelVersion; onClose: () => void; onChanged: () => void;
}) {
  const { notify } = useStore();
  const { confirm } = useDialogs();
  const [formats, setFormats] = useState<ExportFormat[]>([]);
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState("");
  const [result, setResult] = useState("");
  const slug = model.name.toLowerCase().replace(/[^a-z0-9._-]+/g, "-");
  const [repo, setRepo] = useState(slug);
  const [isPrivate, setPrivate] = useState(true);
  useEffect(() => {
    api.modelFormats(model.name, model.version).then((r) => {
      setFormats(r.formats);
      setChosen(new Set(r.formats.filter((f) => f.suggested).map((f) => f.id)));
    }).catch(() => undefined);
  }, [model.name, model.version]);
  const act = async (label: string, fn: () => Promise<string>) => {
    setBusy(label);
    setResult("");
    try {
      setResult(await fn());
      onChanged();
    } catch (e) {
      notify(String((e as Error).message), true);
    } finally {
      setBusy("");
    }
  };
  const mirror = (to: string) => act(to, async () => {
    const v = await api.mirrorModel(model.name, model.version, [to]);
    const m = v.mirrors?.find((x) => x.target === to);
    return m ? `Registered in ${to === "mlflow" ? "MLflow" : "W&B"}: ${m.name} v${m.version}`
      : "Mirrored";
  });
  const publish = async (kind: "hub" | "space") => {
    const id = repo.trim();
    if (!id) return;
    const what = kind === "hub" ? "model repository" : "Space (a Docker app serving /predict)";
    const ok = await confirm(`Publish ${model.name} v${model.version} to the Hugging Face ${what} ` +
                 `“${id}” (${isPrivate ? "private" : "PUBLIC"})?`,
                             { title: "Publish to Hugging Face", ok: "Publish", danger: !isPrivate });
    if (!ok) return;
    act(kind, async () => {
      const r = kind === "hub"
        ? await api.pushToHub(id, model.name, model.version, isPrivate)
        : await api.deploySpace(id, model.name, model.version, isPrivate, [...chosen]);
      return `Published: ${r.url}`;
    });
  };
  return (
    <Modal title={`Share ${model.name} v${model.version}`} onClose={onClose} wide>
      <section>
        <h3>Download</h3>
        <p className="dim">A FastAPI server, the model with its preprocessing, a Dockerfile
          and a README; tick the extra formats to include.</p>
        <div className="form">
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
              </label>);
          })}
        </div>
        <a href={api.modelDeployUrl(model.name, model.version, [...chosen])} download>
          <button className="primary" data-testid="share-download">Download package</button></a>
      </section>
      <section>
        <h3>Your model registries</h3>
        <p className="dim">Registers this version, with its metrics and files, where your
          team already looks. Uses <code>MLFLOW_TRACKING_URI</code> and <code>wandb login</code>.</p>
        <div className="toolbar">
          <button onClick={() => mirror("mlflow")} disabled={!!busy} data-testid="share-mlflow">
            {busy === "mlflow" ? "Registering…" : "Register in MLflow"}</button>
          <button onClick={() => mirror("wandb")} disabled={!!busy} data-testid="share-wandb">
            {busy === "wandb" ? "Logging…" : "Log to W&B"}</button>
        </div>
      </section>
      <section>
        <h3>Hugging Face Hub</h3>
        <p className="dim">Signed in on the server with <code>hf auth login</code>. You are asked
          to confirm before anything is published.</p>
        <div className="toolbar">
          <input value={repo} onChange={(e) => setRepo(e.target.value)} style={{ flex: 1 }}
                 placeholder="owner/name" aria-label="Hub repository" data-testid="share-repo" />
          <label className="check"><input type="checkbox" checked={isPrivate}
            onChange={(e) => setPrivate(e.target.checked)} /> private</label>
        </div>
        <div className="toolbar">
          <button onClick={() => publish("hub")} disabled={!!busy || !repo.trim()}
                  data-testid="share-hub">{busy === "hub" ? "Pushing…" : "Push model"}</button>
          <button onClick={() => publish("space")} disabled={!!busy || !repo.trim()}
                  data-testid="share-space">{busy === "space" ? "Deploying…"
                    : "Deploy as a Space"}</button>
        </div>
      </section>
      {result && <p className="card" data-testid="share-result">{result}</p>}
    </Modal>
  );
}

function PullModal({ onClose, onChanged }: { onClose: () => void; onChanged: () => void }) {
  const { notify } = useStore();
  const [repo, setRepo] = useState("");
  const [busy, setBusy] = useState(false);
  const pull = async () => {
    setBusy(true);
    try {
      const v = await api.pullFromHub(repo.trim());
      notify(`Registered ${v.name} v${v.version} from the Hub`);
      onChanged();
      onClose();
    } catch (e) {
      notify(String((e as Error).message), true);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="Pull a model from the Hub" onClose={onClose} actions={
      <button className="primary" onClick={pull} disabled={busy || !repo.trim()}>
        {busy ? "Downloading…" : "Pull"}</button>}>
      <p className="dim">A repository pushed by AI Made Easy (it has an <code>aime.json</code>)
        becomes a registered version here, ready to serve or to share again.</p>
      <input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="owner/name"
             aria-label="Hub repository" style={{ width: "100%" }} />
    </Modal>
  );
}

export function ModelsPage() {
  const { notify } = useStore();
  const { confirm } = useDialogs();
  const [loaded, setModels] = useState<ModelVersion[] | null>(null);
  const models = loaded ?? [];
  const [error, setError] = useState("");
  const [sharing, setSharing] = useState<ModelVersion | null>(null);
  const [pulling, setPulling] = useState(false);
  const refresh = useCallback(() => {
    api.models().then((r) => { setModels(r.models); setError(""); })
      .catch((e) => setError(String(e.message)));
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
    if (!(await confirm(`Delete ${m.name} v${m.version}?`, { title: "Delete model", ok: "Delete",
                                                             danger: true }))) return;
    await api.deleteModel(m.name, m.version).catch((e) => notify(String(e.message), true));
    refresh();
  };
  return (
    <div className="page" data-testid="models-page">
      <div className="toolbar">
        <h2 style={{ flex: 1 }}>Model registry</h2>
        <button onClick={() => setPulling(true)} data-testid="hub-pull">Pull from the Hub…</button>
      </div>
      <p className="dim">Register finished runs from Experiments ▸ Deploy. Promote a version to
        staging or production, download its serving package, or share it to MLflow, W&B or
        the Hugging Face Hub.</p>
      <table>
        <thead><tr><th>Model</th><th className="num">Version</th><th>Stage</th><th>Framework</th>
          <th>Metrics</th><th>Run</th><th>Registered</th><th /></tr></thead>
        <tbody>
          {models.map((m) => (
            <tr key={`${m.name}-${m.version}`}>
              <td><strong>{m.name}</strong>{m.description && <div className="muted">{m.description}</div>}
                {m.mirrors?.map((x) => (
                  <span key={x.target} className="chip" title={x.uri} data-testid={`mirror-${x.target}`}>
                    {x.target === "mlflow" ? "MLflow" : "W&B"} v{x.version}</span>))}</td>
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
                <a href={api.modelDeployUrl(m.name, m.version, [])} download>
                  <button>Download package</button></a>
                <button onClick={() => setSharing(m)} data-testid={`share-${m.name}-${m.version}`}>
                  Share…</button>
                <button className="danger" onClick={() => remove(m)}>Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <ListState items={loaded} error={error} onRetry={refresh} what="models"
                 empty="No registered models yet: register a finished run from Experiments." />
      {sharing && <ShareModal model={sharing} onClose={() => setSharing(null)} onChanged={refresh} />}
      {pulling && <PullModal onClose={() => setPulling(false)} onChanged={refresh} />}
    </div>
  );
}
