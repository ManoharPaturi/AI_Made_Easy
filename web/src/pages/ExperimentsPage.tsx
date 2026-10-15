import { useState } from "react";

import { useStore } from "../state";
import { PipelinesList } from "../components/Pipeline";

import { Runs } from "./experiments/Runs";
import { Sweeps } from "./experiments/Sweeps";

export { paretoFront } from "./experiments/Sweeps";

export function ExperimentsPage({ initialTab = "runs" }: { initialTab?: string }) {
  const { name } = useStore();
  const [tab, setTab] = useState(initialTab);
  const [scoped, setScoped] = useState(true);
  return (
    <div className="page" data-testid="experiments-page">
      <div className="toolbar" style={{ marginBottom: 10 }}>
        <h2 style={{ margin: 0 }}>Experiments</h2>
        <div className="tabs" style={{ border: "none", padding: 0 }}>
          {["runs", "sweeps", "pipelines"].map((t) => (
            <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
              {t[0].toUpperCase() + t.slice(1)}</button>
          ))}
        </div>
        {tab === "runs" && (
          <label className="dim"><input type="checkbox" checked={scoped}
                                        onChange={(e) => setScoped(e.target.checked)} /> only “{name}”</label>
        )}
      </div>
      {tab === "runs" ? <Runs scope={scoped ? name : ""} /> : tab === "sweeps" ? <Sweeps />
        : <PipelinesList />}
    </div>
  );
}
