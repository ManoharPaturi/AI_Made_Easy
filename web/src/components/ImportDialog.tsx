import { useState } from "react";

import { api } from "../api";
import { useStore } from "../state";
import { Modal } from "./Modal";

export function ImportDialog({ onClose }: { onClose: () => void }) {
  const { load, notify } = useStore();
  const [kind, setKind] = useState("onnx");
  const [file, setFile] = useState<File | null>(null);
  const [attr, setAttr] = useState("");
  const [shape, setShape] = useState("");
  const [report, setReport] = useState("");
  const [busy, setBusy] = useState(false);
  const [graph, setGraph] = useState<Parameters<typeof load>[0] | null>(null);
  const [uri, setUri] = useState("");
  const mlflow = kind === "mlflow";
  const show = (r: { graph: Parameters<typeof load>[0]; summary: string; ok: boolean;
                     warnings: string[] }) => {
    setReport(r.summary + (r.warnings.length ? `\n\nNotes:\n  - ${r.warnings.join("\n  - ")}` : ""));
    setGraph(r.ok ? r.graph : null);
  };
  const run = async () => {
    if (mlflow) {
      setBusy(true);
      setReport("Loading the MLflow model…");
      try {
        show(await api.importMlflow(uri.trim(), shape));
      } catch (e) {
        setReport(`Import failed:\n${(e as Error).message}`);
        setGraph(null);
      } finally {
        setBusy(false);
      }
      return;
    }
    if (!file) return;
    const form = new FormData();
    form.append("kind", kind);
    form.append("file", file);
    form.append("attr", attr);
    form.append("input_shape", shape);
    setBusy(true);
    setReport("Importing…");
    try {
      show(await api.importModel(form));
    } catch (e) {
      setReport(`Import failed:\n${(e as Error).message}`);
      setGraph(null);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="Import model" onClose={onClose} actions={
      <>
        <button onClick={run} disabled={(mlflow ? !uri.trim() : !file) || busy}>Import</button>
        <button className="primary" disabled={!graph} onClick={() => {
          load(graph!);
          notify("Imported model opened as a project");
          onClose();
        }}>Open as project</button>
      </>}>
      <p className="dim">Turns an existing model into editable blocks. The rebuilt model is
        verified against the original (parameter count, outputs).</p>
      <div className="form">
        <div className="field"><label htmlFor="imp-kind">Format</label>
          <select id="imp-kind" value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="onnx">ONNX (.onnx)</option>
            <option value="keras">Keras (.keras / .h5)</option>
            <option value="pytorch">PyTorch module (.py, .pt)</option>
            <option value="mlflow">MLflow model (URI)</option>
          </select></div>
        {mlflow ? <div className="field"><label htmlFor="imp-uri">Model URI</label>
          <input id="imp-uri" value={uri} placeholder="models:/name/1 or runs:/id/model"
                 onChange={(e) => setUri(e.target.value)} /></div>
          : <div className="field"><label htmlFor="imp-file">File</label>
          <input id="imp-file" type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></div>}
        {kind === "pytorch" && <div className="field"><label htmlFor="imp-attr">Model class</label>
          <input id="imp-attr" value={attr} onChange={(e) => setAttr(e.target.value)} /></div>}
        <div className="field"><label htmlFor="imp-shape">Input shape (per sample)</label>
          <input id="imp-shape" placeholder="3, 224, 224" value={shape}
                 onChange={(e) => setShape(e.target.value)} /></div>
      </div>
      {report && <pre className="code" style={{ marginTop: 10, whiteSpace: "pre-wrap" }}>{report}</pre>}
    </Modal>
  );
}
