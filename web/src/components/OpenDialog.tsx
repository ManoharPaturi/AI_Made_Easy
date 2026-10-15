import { useEffect, useState } from "react";

import { api } from "../api";
import { rank, recentProjects, rememberProject } from "../recent";
import { useStore } from "../state";
import type { SavedProject } from "../types";
import { Modal } from "./Modal";

export function OpenDialog({ onClose }: { onClose: () => void }) {
  const { load, notify } = useStore();
  const [samples, setSamples] = useState<string[]>([]);
  const [projects, setProjects] = useState<SavedProject[]>([]);
  useEffect(() => {
    api.samples().then((r) => setSamples(r.samples)).catch(() => undefined);
    api.projects().then((r) => setProjects(r.projects)).catch(() => undefined);
  }, []);
  const recent = recentProjects();
  const open = async (fn: () => Promise<Parameters<typeof load>[0]>, saved?: string) => {
    try {
      load(await fn());
      if (saved) rememberProject(saved);
      onClose();
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const upload = (file: File) => file.text().then((t) => open(async () => JSON.parse(t)));
  return (
    <Modal title="Open" onClose={onClose}>
      <h4>Saved projects</h4>
      {[...projects].sort((a, b) => rank(recent, a.name) - rank(recent, b.name) || b.modified - a.modified)
        .map((p) => (
        <div key={p.name} className="issue" role="button" tabIndex={0}
             onKeyDown={(e) => e.key === "Enter" && open(() => api.project(p.name), p.name)} onClick={() => open(() => api.project(p.name), p.name)}>
          <span style={{ flex: 1 }}>{p.name}</span>
          {p.family && <span className="chip" title="Family">{p.family}</span>}
          {p.task && <span className="chip" title="Task">{p.task}</span>}
          {recent.includes(p.name) && <span className="chip ok" title="Opened recently">recent</span>}
          <span className="muted">{new Date(p.modified * 1000).toLocaleString()}</span>
        </div>
      ))}
      {!projects.length && <p className="muted">No saved projects yet.</p>}
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
