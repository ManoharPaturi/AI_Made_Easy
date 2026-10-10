// New-project wizard: pick a task, point at data (or use demo data), set a budget, then
// pick one of the ranked recipes, or let AutoML search them.
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import { useStore } from "../state";
import type { DataFacts, DeviceProfile, RecipeSuggestion, WizardTask } from "../types";
import { Modal } from "./Modal";

const STEPS = ["Task", "Data", "Budget", "Recipe"];

function count(n: number | undefined): string {
  if (n == null) return "—";
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n);
}

function memory(gb: number): string {
  return gb < 0.1 ? `${Math.max(1, Math.round(gb * 1024))} MB` : `${gb.toFixed(1)} GB`;
}

function TaskStep({ tasks, task, pick }: {
  tasks: WizardTask[]; task: WizardTask | null; pick: (t: WizardTask) => void;
}) {
  const [query, setQuery] = useState("");
  const groups = useMemo(() => {
    const q = query.toLowerCase();
    const out = new Map<string, WizardTask[]>();
    for (const t of tasks) {
      if (q && !`${t.label} ${t.description} ${t.modalities.join(" ")}`.toLowerCase().includes(q))
        continue;
      out.set(t.family_label, [...(out.get(t.family_label) ?? []), t]);
    }
    return [...out.entries()];
  }, [tasks, query]);
  return (
    <>
      <input placeholder="Search tasks (images, forecast, text, …)" value={query} autoFocus
             onChange={(e) => setQuery(e.target.value)}
             style={{ width: "100%", boxSizing: "border-box", marginBottom: 8 }} />
      <div style={{ maxHeight: 380, overflow: "auto" }}>
        {groups.map(([family, rows]) => (
          <div key={family}>
            <h4 style={{ margin: "10px 0 4px" }}>{family}</h4>
            {rows.map((t) => (
              <div key={t.id} className={`issue ${task?.id === t.id ? "selected" : ""}`}
                   data-testid={`task-${t.id}`} onClick={() => pick(t)} style={{ cursor: "pointer" }}>
                <span style={{ flex: 1 }}><b>{t.label}</b>
                  <span className="muted"> — {t.description}</span></span>
                <span className="chip">{t.modalities.join(" · ")}</span>
                {t.automl && <span className="chip ok" title="AutoML can search this task">AutoML</span>}
              </div>
            ))}
          </div>
        ))}
      </div>
    </>
  );
}

function DataStep({ task, facts, setFacts, modality, setModality }: {
  task: WizardTask; facts: DataFacts | null; setFacts: (f: DataFacts | null) => void;
  modality: string; setModality: (m: string) => void;
}) {
  const { notify } = useStore();
  const [own, setOwn] = useState(!!facts);
  const [path, setPath] = useState(facts?.source ?? "");
  const [target, setTarget] = useState(facts?.target ?? "");
  const [busy, setBusy] = useState(false);
  const detect = async () => {
    setBusy(true);
    try {
      setFacts(await api.detect(path, target, task.id));
    } catch (e) {
      notify(String((e as Error).message), true);
      setFacts(null);
    } finally {
      setBusy(false);
    }
  };
  const fits = !facts || facts.tasks.includes(task.id);
  return (
    <>
      <label className="dim"><input type="radio" checked={!own}
                                    onChange={() => { setOwn(false); setFacts(null); }} /> Use demo data
        (each recipe brings a small dataset, no files needed)</label><br />
      <label className="dim"><input type="radio" checked={own} data-testid="own-data"
                                    onChange={() => setOwn(true)} /> Use my data</label>
      {!own && task.demo_modalities.length > 1 && (
        <div className="field" style={{ marginTop: 10 }}>
          <label htmlFor="wiz-modality">Kind of data</label>
          <select id="wiz-modality" data-testid="wizard-modality" value={modality}
                  onChange={(e) => setModality(e.target.value)}>
            {task.demo_modalities.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>)}
      {own && (
        <div style={{ marginTop: 10 }}>
          <div className="field"><label htmlFor="wiz-path">File or folder on the server</label>
            <input id="wiz-path" data-testid="wizard-path" placeholder="/data/churn.csv or /data/images/"
                   value={path} onChange={(e) => setPath(e.target.value)} />
            <div className="help">A table (.csv, .tsv, .parquet, .xlsx), a folder with one sub-folder per
              class, or a COCO / YOLO / VOC / images + masks folder.
              {task.data_kinds.length > 0 && ` This task reads: ${task.data_kinds.join(", ").replace(/_/g, " ")}.`}
            </div></div>
          <div className="field"><label htmlFor="wiz-target">Target column (tables, optional)</label>
            <input id="wiz-target" value={target} onChange={(e) => setTarget(e.target.value)} /></div>
          <button className="primary" disabled={!path || busy} onClick={detect}
                  data-testid="detect">{busy ? "Reading…" : "Detect"}</button>
          {facts && (
            <div style={{ marginTop: 10 }} data-testid="facts">
              <div><b>{facts.summary}</b></div>
              {facts.target && <div className="dim">target: {facts.target}
                {facts.classes.length > 0 && ` · classes: ${facts.classes.slice(0, 8).join(", ")}`}</div>}
              {facts.details.map((d) => <div key={d} className="muted">{d}</div>)}
              {facts.warnings.map((w) => <div key={w} className="sev-warning">⚠ {w}</div>)}
              {!fits && <div className="sev-warning">This data looks like
                {" "}{facts.tasks.join(" / ").replace(/_/g, " ") || "another task"}; recipes for
                {" "}{task.label} may not read it.</div>}
            </div>)}
        </div>)}
    </>
  );
}

function BudgetStep({ budget, setBudget }: {
  budget: Record<string, string>; setBudget: (b: Record<string, string>) => void;
}) {
  const [devices, setDevices] = useState<DeviceProfile[]>([]);
  useEffect(() => { api.devices().then((r) => setDevices(r.devices)).catch(() => undefined); }, []);
  const field = (key: string, label: string, help: string) => (
    <div className="field"><label htmlFor={`wiz-${key}`}>{label}</label>
      <input id={`wiz-${key}`} type="number" min={0} step="any" value={budget[key] ?? ""}
             onChange={(e) => setBudget({ ...budget, [key]: e.target.value })} />
      <div className="help">{help}</div></div>);
  return (
    <>
      <p className="dim">Optional. Recipes that would not fit are ranked lower and flagged; AutoML
        skips them.</p>
      <div className="field"><label htmlFor="wiz-device">Target device</label>
        <select id="wiz-device" value={budget.device ?? ""}
                onChange={(e) => setBudget({ ...budget, device: e.target.value })}>
          <option value="">No device</option>
          {devices.map((d) => <option key={d.id} value={d.id}>{d.label} ({d.memory_gb} GB)</option>)}
        </select></div>
      {field("max_train_memory_gb", "Training memory (GB)", "e.g. 8 for a laptop GPU")}
      {field("max_latency_ms", "Inference latency (ms)", "per sample on the device")}
      {field("max_params_m", "Parameters (millions)", "a cap on model size")}
    </>
  );
}

function RecipeStep({ rows, chosen, choose, knobs, setKnobs }: {
  rows: RecipeSuggestion[] | null; chosen: string; choose: (id: string) => void;
  knobs: Record<string, unknown>; setKnobs: (k: Record<string, unknown>) => void;
}) {
  if (!rows) return <div className="empty">Ranking recipes…</div>;
  if (!rows.length) return <div className="empty">No recipe reads this data for this task.</div>;
  const pick = rows.find((r) => r.recipe.id === chosen);
  return (
    <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
      <div style={{ flex: "3 1 340px", maxHeight: 400, overflow: "auto" }} data-testid="recipes">
        {rows.map((r, i) => (
          <div key={r.recipe.id} data-testid={`recipe-${r.recipe.id}`}
               className={`issue ${chosen === r.recipe.id ? "selected" : ""}`}
               onClick={() => choose(r.recipe.id)} style={{ cursor: "pointer", display: "block" }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <b style={{ flex: 1 }}>{i + 1}. {r.recipe.title}</b>
              <span className="chip">{r.recipe.tier_label}</span>
              <span className={`chip ${r.ready ? "ok" : "err"}`}>{r.ready ? "ready" : "not ready"}</span>
            </div>
            <div className="muted">{r.recipe.description}</div>
            {r.reasons.map((x) => <div key={x} className="dim">✓ {x}</div>)}
            {[...r.cautions, ...r.over_budget.map((o) => `over budget: ${o}`)].map((x) => (
              <div key={x} className="sev-warning">! {x}</div>))}
            {r.errors.map((x) => <div key={x} className="sev-error">✗ {x}</div>)}
            {r.estimate.params != null && (
              <div className="muted">{count(r.estimate.params)} parameters
                {r.estimate.train_memory_gb != null && ` · ~${memory(r.estimate.train_memory_gb)} to train`}
                {r.estimate.latency_ms != null && ` · ${r.estimate.latency_ms.toFixed(1)} ms`}</div>)}
          </div>
        ))}
      </div>
      {pick && (
        <div style={{ flex: "2 1 200px" }}>
          <h4 style={{ marginTop: 0 }}>Settings</h4>
          {!pick.recipe.knobs.length && <p className="muted">This recipe has no settings.</p>}
          {pick.recipe.knobs.map((k) => (
            <div className="field" key={k.name}><label htmlFor={`knob-${k.name}`}>{k.name.replace(/_/g, " ")}</label>
              {k.kind === "choice"
                ? <select id={`knob-${k.name}`} value={String(knobs[k.name] ?? k.default)}
                          onChange={(e) => setKnobs({ ...knobs, [k.name]: k.values.find((v) => String(v) === e.target.value) })}>
                    {k.values.map((v) => <option key={String(v)} value={String(v)}>{String(v)}</option>)}
                  </select>
                : <input id={`knob-${k.name}`} type="number" step="any" min={k.low ?? undefined}
                         max={k.high ?? undefined} value={String(knobs[k.name] ?? k.default)}
                         onChange={(e) => setKnobs({ ...knobs, [k.name]: Number(e.target.value) })} />}
              {k.help && <div className="help">{k.help}</div>}
            </div>))}
        </div>)}
    </div>
  );
}

export function NewProjectWizard({ onClose, onCreated, onAutoml }: {
  onClose: () => void; onCreated: () => void; onAutoml: () => void;
}) {
  const { load, setName, notify, name } = useStore();
  const [step, setStep] = useState(0);
  const [tasks, setTasks] = useState<WizardTask[]>([]);
  const [task, setTask] = useState<WizardTask | null>(null);
  const [facts, setFacts] = useState<DataFacts | null>(null);
  const [modality, setModality] = useState("");
  const [budget, setBudget] = useState<Record<string, string>>({});
  const [rows, setRows] = useState<RecipeSuggestion[] | null>(null);
  const [chosen, setChosen] = useState("");
  const [knobs, setKnobs] = useState<Record<string, unknown>>({});
  const [trials, setTrials] = useState(12);
  const [epochs, setEpochs] = useState(0);
  useEffect(() => {
    api.wizardTasks().then((r) => setTasks(r.tasks)).catch((e) => notify(String(e.message), true));
  }, [notify]);
  const budgetValues = useMemo(() => Object.fromEntries(Object.entries(budget)
    .filter(([, v]) => v !== "" && v != null)
    .map(([k, v]) => [k, k === "device" ? v : Number(v)])), [budget]);
  useEffect(() => {
    if (step !== 3 || !task) return;
    setRows(null);
    api.recommend({ task: task.id, facts, budget: budgetValues, modality: facts ? "" : modality })
      .then((r) => {
        setRows(r.suggestions);
        const first = r.suggestions.find((s) => s.ready) ?? r.suggestions[0];
        setChosen(first?.recipe.id ?? "");
        setKnobs({});
      })
      .catch((e) => { notify(String(e.message), true); setRows([]); });
  }, [step, task, facts, budgetValues, modality, notify]);

  const pickTask = (t: WizardTask) => {
    setTask(t);
    setModality(t.demo_modalities[0] ?? "");
    setFacts(null);
  };
  const create = async () => {
    const pick = rows?.find((r) => r.recipe.id === chosen);
    if (!pick || !task) return;
    try {
      const graph = Object.keys(knobs).length || !pick.graph
        ? (await api.buildRecipe({ recipe: chosen, task: task.id, facts, knobs,
                                   budget: budgetValues })).graph
        : pick.graph;
      load(graph);
      setName(graph.name || task.id);
      notify(`Created a ${pick.recipe.title} design — see “Why” for what each block does`);
      onCreated();
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const automl = async () => {
    if (!task) return;
    try {
      const r = await api.startAutoml({ task: task.id, facts: facts ?? {}, modality: facts ? "" : modality,
                                        budget: budgetValues, max_trials: trials, epochs }, name);
      notify(`AutoML started: ${r.candidates.length} recipes, optimising ${r.metric}`);
      onAutoml();
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const canNext = step === 0 ? !!task : step === 1 ? true : step === 2;
  const pick = rows?.find((r) => r.recipe.id === chosen);
  return (
    <Modal wide title={`New project — ${STEPS[step]}${task && step ? `: ${task.label}` : ""}`} onClose={onClose}
           actions={<>
             {step > 0 && <button onClick={() => setStep(step - 1)}>Back</button>}
             {step < 3 && <button className="primary" disabled={!canNext} data-testid="wizard-next"
                                  onClick={() => setStep(step + 1)}>Next</button>}
             {step === 3 && task?.automl && (
               <span className="toolbar">
                 <label className="dim">trials <input type="number" min={1} value={trials} style={{ width: 54 }}
                                                      onChange={(e) => setTrials(Number(e.target.value))} /></label>
                 <label className="dim">epochs <input type="number" min={0} value={epochs} style={{ width: 54 }}
                                                      title="0: each recipe's default"
                                                      onChange={(e) => setEpochs(Number(e.target.value))} /></label>
                 <button onClick={automl} data-testid="start-automl"
                         disabled={!rows?.some((r) => r.ready)}>Run AutoML</button>
               </span>)}
             {step === 3 && <button className="primary" onClick={create} data-testid="create-design"
                                    disabled={!pick || !!pick.errors.length}>Create design</button>}
           </>}>
      <div className="tabs" style={{ marginBottom: 10 }}>
        {STEPS.map((s, i) => (
          <button key={s} className={step === i ? "active" : ""} disabled={i > 0 && !task}
                  onClick={() => setStep(i)}>{i + 1}. {s}</button>))}
      </div>
      <div data-testid="wizard">
        {step === 0 && <TaskStep tasks={tasks} task={task} pick={pickTask} />}
        {step === 1 && task && <DataStep task={task} facts={facts} setFacts={setFacts}
                                         modality={modality} setModality={setModality} />}
        {step === 2 && <BudgetStep budget={budget} setBudget={setBudget} />}
        {step === 3 && <RecipeStep rows={rows} chosen={chosen} knobs={knobs} setKnobs={setKnobs}
                                   choose={(id) => { setChosen(id); setKnobs({}); }} />}
      </div>
    </Modal>
  );
}
