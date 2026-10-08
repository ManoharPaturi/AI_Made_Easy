import { useEffect, useState } from "react";

import { api } from "./api";
import { Modal } from "./components/Modal";
import { Designer } from "./designer/Designer";
import { DataPage } from "./pages/DataPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { ModelsPage } from "./pages/ModelsPage";
import { useStore } from "./state";

const PAGES = [["design", "Design"], ["data", "Data"], ["experiments", "Experiments"],
               ["models", "Models"]] as const;
type Page = (typeof PAGES)[number][0];

function OpenDialog({ onClose }: { onClose: () => void }) {
  const { load, notify } = useStore();
  const [samples, setSamples] = useState<string[]>([]);
  const [projects, setProjects] = useState<{ name: string; modified: number }[]>([]);
  useEffect(() => {
    api.samples().then((r) => setSamples(r.samples)).catch(() => undefined);
    api.projects().then((r) => setProjects(r.projects)).catch(() => undefined);
  }, []);
  const open = async (fn: () => Promise<Parameters<typeof load>[0]>) => {
    try {
      load(await fn());
      onClose();
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const upload = (file: File) => file.text().then((t) => open(async () => JSON.parse(t)));
  return (
    <Modal title="Open" onClose={onClose}>
      <h4>Saved projects</h4>
      {projects.length ? projects.map((p) => (
        <div key={p.name} className="issue" onClick={() => open(() => api.project(p.name))}>
          <span style={{ flex: 1 }}>{p.name}</span>
          <span className="muted">{new Date(p.modified * 1000).toLocaleString()}</span>
        </div>
      )) : <p className="muted">No saved projects yet.</p>}
      <h4>Examples</h4>
      <div className="cards">
        {samples.map((s) => (
          <button key={s} onClick={() => open(() => api.sample(s))} data-testid={`sample-${s}`}>
            {s.replace(/\.json$/, "").replace(/_/g, " ")}</button>
        ))}
      </div>
      <h4>From your computer</h4>
      <input type="file" accept=".json,.aime" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
    </Modal>
  );
}

function ImportDialog({ onClose }: { onClose: () => void }) {
  const { load, notify } = useStore();
  const [kind, setKind] = useState("onnx");
  const [file, setFile] = useState<File | null>(null);
  const [attr, setAttr] = useState("");
  const [shape, setShape] = useState("");
  const [report, setReport] = useState("");
  const [busy, setBusy] = useState(false);
  const [graph, setGraph] = useState<Parameters<typeof load>[0] | null>(null);
  const run = async () => {
    if (!file) return;
    const form = new FormData();
    form.append("kind", kind);
    form.append("file", file);
    form.append("attr", attr);
    form.append("input_shape", shape);
    setBusy(true);
    setReport("Importing…");
    try {
      const r = await api.importModel(form);
      setReport(r.summary + (r.warnings.length ? `\n\nNotes:\n  - ${r.warnings.join("\n  - ")}` : ""));
      setGraph(r.ok ? r.graph : null);
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
        <button onClick={run} disabled={!file || busy}>Import</button>
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
            <option value="pytorch">PyTorch module (.py)</option>
          </select></div>
        <div className="field"><label htmlFor="imp-file">File</label>
          <input id="imp-file" type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></div>
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

export function App() {
  const { name, setName, project, notify, toast, theme, toggleTheme, version, validation, nodes,
          dataIssues } = useStore();
  const [page, setPage] = useState<Page>("design");
  const [dialog, setDialog] = useState<"" | "open" | "import">("");

  const save = async () => {
    try {
      await api.saveProject(name, project());
      notify(`Saved “${name}”`);
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const download = () => {
    const blob = new Blob([JSON.stringify(project(), null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name || "project"}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        save();
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });

  const errors = validation?.issues.filter((i) => i.severity === "error").length ?? 0;
  const warnings = (validation?.issues.length ?? 0) - errors + dataIssues.length;
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">AI <span>Made Easy</span></div>
        <nav className="nav">
          {PAGES.map(([id, label]) => (
            <button key={id} className={page === id ? "active" : ""} onClick={() => setPage(id)}>
              {label}</button>
          ))}
        </nav>
        <div className="spacer" />
        <input className="project-name" value={name} onChange={(e) => setName(e.target.value)}
               aria-label="Project name" />
        <button onClick={() => setDialog("open")}>Open</button>
        <button onClick={save} title="Save on the server (Ctrl+S)">Save</button>
        <button onClick={download} title="Download the project JSON">Download</button>
        <button onClick={() => setDialog("import")}>Import model</button>
        <button className="ghost" onClick={toggleTheme} title="Toggle theme">
          {theme === "dark" ? "☀" : "☾"}</button>
      </header>
      {page === "design" && <Designer />}
      {page === "data" && <DataPage />}
      {page === "experiments" && <ExperimentsPage />}
      {page === "models" && <ModelsPage />}
      <footer className="statusbar">
        <span className={errors ? "sev-error" : warnings ? "sev-warning" : "sev-ok"} data-testid="status">
          {!nodes.length ? "Empty design" : errors ? `${errors} error(s)` : warnings
            ? `${warnings} warning(s)` : "No problems"}</span>
        <span>{nodes.length} blocks</span>
        <span className="spacer" />
        <span>AI Made Easy {version}</span>
      </footer>
      {dialog === "open" && <OpenDialog onClose={() => setDialog("")} />}
      {dialog === "import" && <ImportDialog onClose={() => setDialog("")} />}
      {toast && <div className={`toast ${toast.error ? "error" : ""}`} role="status">{toast.text}</div>}
    </div>
  );
}
