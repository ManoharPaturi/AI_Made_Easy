import { useEffect, useState } from "react";

import { api } from "../api";
import { metric } from "../components/Modal";
import { useStore } from "../state";
import type { DataProfile, SplitPreview } from "../types";

const SEV = { error: "✕", warning: "⚠", info: "ℹ" } as const;

export function DataPage() {
  const { project, nodes, blockMap, notify } = useStore();
  const datasets = nodes.filter((n) => n.data.typeId.startsWith("data."));
  const [nodeId, setNodeId] = useState<string>("");
  const [profile, setProfile] = useState<DataProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [split, setSplit] = useState<SplitPreview | null>(null);
  const [augment, setAugment] = useState<Awaited<ReturnType<typeof api.augment>> | null>(null);
  const [busy, setBusy] = useState("");

  const current = nodeId || datasets[0]?.id || "";
  const params = JSON.stringify(datasets.find((n) => n.id === current)?.data.params ?? {});

  useEffect(() => {
    if (!current) return;
    setLoading(true);
    setSplit(null);
    setAugment(null);
    api.profile(project(), current).then(setProfile)
      .catch((e) => { setProfile(null); notify(String(e.message), true); })
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, params]);

  if (!datasets.length) {
    return <div className="page"><h2>Data</h2>
      <div className="empty">Add a dataset block (Data category) to the design to profile it.</div>
    </div>;
  }
  const total = profile?.classes.reduce((a, c) => a + c.count, 0) ?? 0;
  const run = async (kind: "split" | "augment") => {
    setBusy(kind);
    try {
      if (kind === "split") setSplit(await api.split(project()));
      else setAugment(await api.augment(project()));
    } catch (e) {
      notify(String((e as Error).message), true);
    } finally {
      setBusy("");
    }
  };
  return (
    <div className="page" data-testid="data-page">
      <div className="toolbar">
        <h2 style={{ margin: 0 }}>Data</h2>
        <select value={current} onChange={(e) => setNodeId(e.target.value)} aria-label="Dataset">
          {datasets.map((n) => (
            <option key={n.id} value={n.id}>
              {blockMap.get(n.data.typeId)?.display_name ?? n.data.typeId} · {n.id}
            </option>
          ))}
        </select>
        <button onClick={() => run("split")} disabled={!!busy}>
          {busy === "split" ? "Computing…" : "Preview Split"}</button>
        <button onClick={() => run("augment")} disabled={!!busy}>
          {busy === "augment" ? "Rendering…" : "Preview Augmentation"}</button>
      </div>
      {loading && <p className="muted">Profiling…</p>}
      {profile && (
        <>
          <p><strong>{profile.summary || profile.source}</strong>
            {profile.error && <span className="sev-error"> {profile.error}</span>}</p>
          {profile.details.map((d, i) => <div key={i} className="dim">{d}</div>)}
          <div className="split" style={{ marginTop: 12 }}>
            <div>
              <h3>Findings</h3>
              {profile.findings.length ? profile.findings.map((f, i) => (
                <div key={i} className="issue" style={{ cursor: "default" }}>
                  <span className={`sev-${f.severity}`}>{SEV[f.severity]}</span>
                  <span>{f.message}{f.hint && <div className="muted">{f.hint}</div>}</span>
                </div>
              )) : ["table", "folder", "arrays", "records"].includes(profile.kind)
                ? <div className="sev-ok">✓ No problems found</div>
                : <p className="muted">Built-in and remote datasets are not profiled.</p>}
            </div>
            <div>
              <h3>Class distribution</h3>
              {profile.classes.length ? (
                <table>
                  <tbody>
                    {profile.classes.map((c) => (
                      <tr key={c.name}><td>{c.name}</td><td className="num">{c.count.toLocaleString()}</td>
                        <td style={{ width: "50%" }}><div className="bar">
                          <div style={{ width: `${(100 * c.count) / Math.max(total, 1)}%` }} /></div></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : <p className="muted">{profile.task === "regression" ? "Regression target" : "—"}</p>}
            </div>
          </div>
          {profile.columns.length > 0 && (
            <>
              <h3>Columns</h3>
              <table data-testid="columns">
                <thead><tr><th>Column</th><th>Role</th><th>Type</th><th className="num">Missing</th>
                  <th className="num">Unique</th><th className="num">Mean</th><th className="num">Std</th>
                  <th className="num">Min</th><th className="num">Max</th><th>Top values</th></tr></thead>
                <tbody>
                  {profile.columns.map((c) => (
                    <tr key={c.name}><td>{c.name}</td><td>{c.role}</td><td>{c.kind}</td>
                      <td className="num">{((100 * c.missing) / Math.max(c.count + c.missing, 1)).toFixed(1)}%</td>
                      <td className="num">{c.unique.toLocaleString()}</td>
                      <td className="num">{metric(c.mean)}</td><td className="num">{metric(c.std)}</td>
                      <td className="num">{metric(c.min)}</td><td className="num">{metric(c.max)}</td>
                      <td className="dim">{c.top.slice(0, 3).map(([v, n]) => `${v} (${n})`).join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
      {split && (
        <>
          <h3>Split — {split.method}</h3>
          <table style={{ maxWidth: 560 }} data-testid="split">
            <thead><tr><th>Class</th><th className="num">Train</th><th className="num">Validation</th>
              <th className="num">Test</th></tr></thead>
            <tbody>
              {Object.entries(split.per_class).map(([cls, c]) => (
                <tr key={cls}><td>{cls}</td><td className="num">{c.train}</td>
                  <td className="num">{c.val}</td><td className="num">{c.test}</td></tr>
              ))}
              <tr><td><strong>Total</strong></td><td className="num"><strong>{split.totals.train}</strong></td>
                <td className="num"><strong>{split.totals.val}</strong></td>
                <td className="num"><strong>{split.totals.test}</strong></td></tr>
            </tbody>
          </table>
          {split.notes.map((n, i) => <div key={i} className="sev-warning">⚠ {n}</div>)}
        </>
      )}
      {augment && (
        <>
          <h3>Augmentation</h3>
          <p className="dim">{augment.train_transforms.length
            ? `Training transforms: ${augment.train_transforms.map((t) => t.split("(")[0].replace("v2.", "")).join(", ")}`
            : "No training augmentation in the design."}
            {augment.skipped.length > 0 && ` · not previewed: ${augment.skipped.join(", ")}`}</p>
          {augment.images.map((img) => (
            <div key={img.source} className="thumbs" style={{ marginBottom: 12 }}>
              {[["original", img.original], ["evaluation", img.eval],
                ...img.train.map((t, i) => [`training ${i + 1}`, t])].map(([label, src]) => (
                <figure key={label}><img src={src} alt={label} /><figcaption>{label}</figcaption></figure>
              ))}
            </div>
          ))}
        </>
      )}
    </div>
  );
}
