// Shapes of the AI Made Easy REST API (see ai_made_easy/server/app.py).

export interface ParamSpec {
  name: string;
  type: "int" | "float" | "bool" | "str" | "enum" | "shape" | string;
  default: unknown;
  options?: string[];
  minimum?: number | null;
  maximum?: number | null;
  help?: string;
}

export interface PortSpec {
  name: string;
  dtype?: string;
}

export interface BlockDef {
  type_id: string;
  display_name: string;
  category: string;
  color: string;
  composite: boolean;
  description: string;
  frameworks: string[];
  params: ParamSpec[];
  inputs: PortSpec[];
  outputs: PortSpec[];
}

export interface ProjectNode {
  id: string;
  type: string;
  params: Record<string, unknown>;
  position: [number, number];
}

export interface ProjectEdge {
  from: string; // "node/port"
  to: string;
}

export interface Project {
  name: string;
  nodes: ProjectNode[];
  edges: ProjectEdge[];
  meta?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Issue {
  severity: "error" | "warning";
  node: string | null;
  message: string;
  fix: { label: string; description: string } | null;
}

export interface Validation {
  valid: boolean;
  issues: Issue[];
  shapes: Record<string, number[]>;
}

export interface Summary {
  total_params: number;
  total_params_display: string;
  layers: { name: string; type: string; output_shape: number[]; params: number }[];
}

export interface RunRecord {
  run_id: string;
  name: string;
  project: string;
  framework: string;
  kind: string;
  status: string;
  created_at: number;
  finished_at: number | null;
  final_metrics: Record<string, number>;
  best_metrics: Record<string, number>;
  epochs_done: number;
  tags: string[];
  note: string;
  parent: string;
  trial: Record<string, unknown>;
  error: string;
  data_fingerprint: string;
  params?: Record<string, unknown>;
  graph?: Project;
  duration?: number | null;
}

export interface EpochEvent {
  type: "epoch";
  epoch: number;
  total?: number;
  metrics: Record<string, number>;
}

export interface RunEvent {
  type: "log" | "epoch" | "error" | "done" | "env" | "status";
  [key: string]: unknown;
}

export interface SweepParam {
  node: string;
  param: string;
  key: string;
  block: string;
  current: unknown;
  kind: "int" | "float" | "choice";
  low?: number;
  high?: number;
  log?: boolean;
  values?: unknown[];
}

export interface SweepDimension {
  node: string;
  param: string;
  kind: string;
  low?: number | null;
  high?: number | null;
  log?: boolean;
  values?: unknown[];
}

export interface SweepTrial {
  number: number;
  values: Record<string, unknown>;
  run_id: string;
  state: string;
  score: number | null;
  message: string;
}

export interface SweepRecord {
  sweep_id: string;
  name: string;
  spec: { dimensions: SweepDimension[]; metric: string; direction: string; strategy: string;
          max_trials: number };
  project: string;
  state: string;
  created_at: number;
  finished_at: number | null;
  trials: SweepTrial[];
  best: { number: number; run_id: string; score: number; values: Record<string, unknown> } | null;
  message: string;
}

export interface Finding {
  severity: "error" | "warning" | "info";
  message: string;
  hint: string;
  classes: string[];
}

export interface ColumnProfile {
  name: string;
  kind: string;
  dtype: string;
  count: number;
  missing: number;
  unique: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  median: number | null;
  max: number | null;
  top: [string, number][];
  histogram: number[];
  role: string;
}

export interface DataProfile {
  type_id: string;
  kind: string;
  summary: string;
  source: string;
  rows: number;
  truncated: boolean;
  columns: ColumnProfile[];
  classes: { name: string; count: number }[];
  task: string;
  findings: Finding[];
  details: string[];
  error: string;
  ok: boolean;
}

export interface SplitPreview {
  totals: Record<string, number>;
  per_class: Record<string, Record<string, number>>;
  method: string;
  notes: string[];
}

export interface ModelVersion {
  name: string;
  version: number;
  run_id: string;
  framework: string;
  task: string;
  stage: string;
  metrics: Record<string, number>;
  description: string;
  created_at: number;
}
