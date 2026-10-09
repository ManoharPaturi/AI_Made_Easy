// Typed client for the AI Made Easy server. Every call goes to /api; errors carry the
// server's user-facing message.
import type {
  BlockDef, BudgetReport, TableLayout, RunSamples, DataProfile, DeviceProfile, FamilyInfo, TaskInfo, ModelVersion, Project, RunEvent, RunRecord, SplitPreview, Summary,
  SweepParam, SweepRecord, Validation,
} from "./types";

export class ApiError extends Error {}

const TOKEN_KEY = "aime.token";

function headers(json: boolean): HeadersInit {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const form = body instanceof FormData;
  const res = await fetch(`/api${path}`, {
    method,
    headers: headers(body !== undefined && !form),
    body: body === undefined ? undefined : form ? body : JSON.stringify(body),
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      message = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* keep the status line */
    }
    throw new ApiError(message);
  }
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("json") ? res.json() : res.text()) as Promise<T>;
}

const get = <T>(path: string) => request<T>("GET", path);
const post = <T>(path: string, body?: unknown) => request<T>("POST", path, body ?? {});

export const api = {
  setToken: (token: string) => localStorage.setItem(TOKEN_KEY, token),
  info: () => get<{ version: string; auth: boolean; blocks: number }>("/info"),

  blocks: () => get<{ blocks: BlockDef[] }>("/blocks"),
  samples: () => get<{ samples: string[] }>("/samples"),
  tasks: () => get<{ tasks: TaskInfo[] }>("/tasks"),
  families: () => get<{ families: FamilyInfo[] }>("/families"),
  describe: (graph: Project) =>
    post<{ family: FamilyInfo; task: TaskInfo | null }>("/describe", { graph }),
  sample: (name: string) => get<Project>(`/samples/${encodeURIComponent(name)}`),

  validate: (graph: Project) => post<Validation>("/validate", { graph }),
  fix: (graph: Project, index: number) =>
    post<{ label: string; graph: Project }>("/fix", { graph, index }),
  targets: () => get<{ targets: { id: string; label: string }[] }>("/targets"),
  generate: (graph: Project, target: string) =>
    post<{ code: string }>("/generate", { graph, target }),
  summary: (graph: Project) => post<Summary>("/summary", { graph }),
  devices: () => get<{ devices: DeviceProfile[] }>("/devices"),
  budget: (graph: Project) => post<BudgetReport>("/budget", { graph }),
  tableLayout: (graph: Project, node: string) =>
    post<TableLayout>("/table_layout", { graph, node }),
  expand: (graph: Project, nodeId: string) =>
    post<{ graph: Project }>("/expand", { graph, node_id: nodeId }),

  projects: () => get<{ projects: { name: string; modified: number }[] }>("/projects"),
  project: (name: string) => get<Project>(`/projects/${encodeURIComponent(name)}`),
  saveProject: (name: string, graph: Project) =>
    request<{ saved: string }>("PUT", `/projects/${encodeURIComponent(name)}`, graph),
  deleteProject: (name: string) =>
    request<{ deleted: string }>("DELETE", `/projects/${encodeURIComponent(name)}`),

  train: (graph: Project, project: string, framework = "auto") =>
    post<{ run_id: string }>("/runs", { graph, project, framework }),
  runs: (project?: string) =>
    get<{ runs: RunRecord[] }>(`/runs${project ? `?project=${encodeURIComponent(project)}` : ""}`),
  run: (id: string) => get<RunRecord>(`/runs/${id}`),
  runSamples: (id: string) => get<RunSamples>(`/runs/${id}/samples`),
  metrics: (id: string) => get<{ epochs: { epoch: number; metrics: Record<string, number> }[] }>(
    `/runs/${id}/metrics`),
  stopRun: (id: string) => post<Record<string, unknown>>(`/runs/${id}/stop`),
  deleteRun: (id: string) => request<{ deleted: string }>("DELETE", `/runs/${id}`),
  tagRun: (id: string, tags: string[], note?: string) =>
    request<RunRecord>("PATCH", `/runs/${id}`, { tags, note }),
  compare: (ids: string[]) => post<Record<string, unknown>>("/runs/compare", { run_ids: ids }),

  sweepParams: (graph: Project) => post<{ params: SweepParam[] }>("/sweeps/params", { graph }),
  startSweep: (graph: Project, spec: Record<string, unknown>, project: string) =>
    post<{ sweep_id: string }>("/sweeps", { graph, spec, project }),
  sweeps: () => get<{ sweeps: SweepRecord[] }>("/sweeps"),
  sweep: (id: string) => get<SweepRecord>(`/sweeps/${id}`),
  stopSweep: (id: string) => post<Record<string, unknown>>(`/sweeps/${id}/stop`),
  bestGraph: (id: string) => get<{ graph: Project }>(`/sweeps/${id}/best`),

  formats: (runId: string) =>
    get<{ formats: { id: string; label: string }[] }>(`/runs/${runId}/formats`),
  deployUrl: (runId: string, formats: string[]) =>
    `/api/runs/${runId}/deploy?formats=${encodeURIComponent(formats.join(","))}`,
  models: () => get<{ models: ModelVersion[] }>("/models"),
  registerModel: (runId: string, name: string, description = "") =>
    post<ModelVersion>("/models", { run_id: runId, name, description }),
  setStage: (name: string, version: number, stage: string) =>
    request<ModelVersion>("PATCH", `/models/${encodeURIComponent(name)}/${version}`, { stage }),
  deleteModel: (name: string, version: number) =>
    request<Record<string, unknown>>("DELETE", `/models/${encodeURIComponent(name)}/${version}`),

  importModel: (form: FormData) =>
    post<{ graph: Project; summary: string; ok: boolean; warnings: string[];
           unsupported: string[] }>("/import", form),

  profile: (graph: Project, nodeId?: string) =>
    post<DataProfile>("/data/profile", { graph, node_id: nodeId ?? null }),
  dataIssues: (graph: Project) =>
    post<{ issues: { severity: string; message: string; node_id: string }[] }>(
      "/data/issues", { graph }),
  split: (graph: Project) => post<SplitPreview>("/data/split", { graph }),
  augment: (graph: Project) =>
    post<{ images: { source: string; original: string; eval: string; train: string[] }[];
           train_transforms: string[]; skipped: string[] }>("/data/augment", { graph }),
};

/** Live run events: a WebSocket that replays the epochs so far, then streams new events.
 *  Falls back to polling when WebSockets are unavailable (e.g. behind some proxies). */
export function streamRun(runId: string, onEvent: (event: RunEvent) => void): () => void {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const token = localStorage.getItem(TOKEN_KEY);
  const url = `${proto}://${location.host}/api/runs/${runId}/events` +
    (token ? `?token=${encodeURIComponent(token)}` : "");
  let closed = false;
  let received = false;
  let timer = 0;
  const poll = async () => {
    if (closed) return;
    try {
      const [run, metrics] = await Promise.all([api.run(runId), api.metrics(runId)]);
      for (const e of metrics.epochs) onEvent({ type: "epoch", ...e } as RunEvent);
      if (run.status !== "running") {
        onEvent({ type: "status", status: run.status, final_metrics: run.final_metrics });
        return;
      }
    } catch {
      /* retry */
    }
    timer = window.setTimeout(poll, 1500);
  };
  const socket = new WebSocket(url);
  socket.onmessage = (msg) => {
    received = true;
    onEvent(JSON.parse(msg.data) as RunEvent);
  };
  socket.onclose = () => {
    if (!received && !closed) poll();
  };
  return () => {
    closed = true;
    window.clearTimeout(timer);
    socket.close();
  };
}
