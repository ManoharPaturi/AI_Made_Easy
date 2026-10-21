// Try a finished run (3.5): a form built from what the model takes, answered by the run's
// own predictor on the server's machine. Nothing is sent anywhere else.
import { useEffect, useState } from "react";

import { api } from "../../api";
import { Modal } from "../../components/Modal";
import type { RunRecord, TryForm } from "../../types";

type Prediction = Record<string, unknown>;

const FILE_KINDS: Record<string, { accept: string; what: string }> = {
  images: { accept: "image/*", what: "an image" },
  audio: { accept: ".wav,audio/wav", what: "a WAV clip" },
  speech: { accept: ".wav,audio/wav", what: "a WAV clip" },
};
const TEXT_KINDS = new Set(["texts", "prompt"]);

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/** A typed value from a field: numbers stay numbers, everything else is text. */
function value(raw: string): string | number {
  const n = Number(raw);
  return raw.trim() !== "" && Number.isFinite(n) ? n : raw;
}

export function TryResult({ prediction }: { prediction: Prediction }) {
  const probs = prediction.probabilities as Record<string, number> | undefined;
  if (probs && typeof probs === "object") {
    const top = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 5);
    return (
      <div className="try-result" data-testid="try-result">
        <div><strong>{String(prediction.label)}</strong>
          {typeof prediction.confidence === "number"
            && <span className="muted"> · {(prediction.confidence * 100).toFixed(1)}% sure</span>}</div>
        {top.map(([name, p]) => (
          <div key={name} className="try-bar">
            <span>{name}</span>
            <span className="bar"><span style={{ width: `${Math.max(1, p * 100)}%` }} /></span>
            <span className="num">{(p * 100).toFixed(1)}%</span>
          </div>))}
      </div>);
  }
  const images = Object.entries(prediction).filter(([, v]) => typeof v === "string"
    && (v as string).length > 200 && /^[A-Za-z0-9+/=]+$/.test((v as string).slice(0, 100)));
  return (
    <div className="try-result" data-testid="try-result">
      {images.map(([k, v]) => <img key={k} alt={k} src={`data:image/png;base64,${v}`} />)}
      <pre className="code">{JSON.stringify(Object.fromEntries(Object.entries(prediction)
        .filter(([k]) => !images.some(([i]) => i === k))), null, 1)}</pre>
    </div>);
}

export function TryModal({ run, onClose }: { run: RunRecord; onClose: () => void }) {
  const [form, setForm] = useState<TryForm | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [fields, setFields] = useState<Record<string, string>>({});
  const [text, setText] = useState("");
  const [json, setJson] = useState("");
  const [file, setFile] = useState<{ name: string; url: string } | null>(null);
  const [result, setResult] = useState<{ predictions: Prediction[]; latency_ms: number } | null>(null);

  useEffect(() => {
    api.tryForm(run.run_id).then((f) => {
      setForm(f);
      setJson(JSON.stringify(f.example ?? [], null, 1));
    }).catch((e) => setError(String((e as Error).message)));
  }, [run.run_id]);

  const kind = form?.input_kind ?? "";
  const columns = kind === "records" ? form?.signature.columns ?? [] : [];
  const fileKind = FILE_KINDS[kind];

  const inputs = (): unknown[] => {
    if (columns.length) return [Object.fromEntries(columns.map((c) => [c, value(fields[c] ?? "")]))];
    if (TEXT_KINDS.has(kind)) return [text];
    if (fileKind) return file ? [file.url] : [];
    const parsed = JSON.parse(json || "[]");
    return Array.isArray(parsed) ? parsed : [parsed];
  };

  const predict = async () => {
    setBusy(true);
    setError("");
    try {
      setResult(await api.tryRun(run.run_id, inputs()));
    } catch (e) {
      setError(String((e as Error).message));
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title={`Try run ${run.run_id}`} onClose={onClose} wide actions={
      <button className="primary" onClick={predict} disabled={!form || busy}
              data-testid="try-predict">{busy ? "Predicting…" : "Predict"}</button>}>
      <p className="dim">The trained model answers here, on the machine running AI Made Easy; nothing
        is sent anywhere else.</p>
      {!form && !error && <p className="muted">Loading the model…</p>}
      {form && (
        <div className="form" data-testid="try-form" data-kind={kind}>
          {columns.length > 0 && columns.map((c) => (
            <div key={c} className="field"><label htmlFor={`try-${c}`}>{c}</label>
              <input id={`try-${c}`} value={fields[c] ?? ""}
                     onChange={(e) => setFields((f) => ({ ...f, [c]: e.target.value }))} /></div>))}
          {TEXT_KINDS.has(kind) && (
            <div className="field"><label htmlFor="try-text">Text</label>
              <textarea id="try-text" rows={4} value={text} onChange={(e) => setText(e.target.value)}
                        placeholder={kind === "prompt" ? "Start a sentence…" : "Write a text to classify…"} /></div>)}
          {fileKind && (
            <div className="field"><label htmlFor="try-file">Choose {fileKind.what}</label>
              <input id="try-file" type="file" accept={fileKind.accept} data-testid="try-file"
                     onChange={async (e) => {
                       const f = e.target.files?.[0];
                       setFile(f ? { name: f.name, url: await readAsDataUrl(f) } : null);
                     }} />
              {file && kind === "images" && <img className="try-preview" src={file.url} alt={file.name} />}</div>)}
          {!columns.length && !TEXT_KINDS.has(kind) && !fileKind && (
            <div className="field"><label htmlFor="try-json">Inputs (JSON list)</label>
              <textarea id="try-json" rows={6} className="code" value={json}
                        onChange={(e) => setJson(e.target.value)} data-testid="try-json" />
              <div className="help">{form.input_help}</div></div>)}
        </div>)}
      {error && <p className="error" role="alert">{error}</p>}
      {result && (
        <div data-testid="try-results">
          {result.predictions.map((p, i) => <TryResult key={i} prediction={p} />)}
          <p className="muted">{result.latency_ms} ms</p>
        </div>)}
    </Modal>
  );
}
