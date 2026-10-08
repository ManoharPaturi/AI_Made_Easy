// App-wide state: the block catalog, the open project (React Flow nodes / edges), live
// validation, undo history and toasts.
import {
  applyEdgeChanges, applyNodeChanges, type Edge, type EdgeChange, type NodeChange,
} from "@xyflow/react";
import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
  type ReactNode,
} from "react";

import { api } from "./api";
import {
  autoLayout, defaultParams, type FlowNode, fromProject, hasOverlaps, newNodeId, structureKey,
  toProject,
} from "./project";
import type { BlockDef, Project, Validation } from "./types";

type Toast = { text: string; error?: boolean } | null;

interface Store {
  blocks: BlockDef[];
  blockMap: Map<string, BlockDef>;
  name: string;
  setName: (name: string) => void;
  nodes: FlowNode[];
  edges: Edge[];
  onNodesChange: (changes: NodeChange<FlowNode>[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  setEdges: (fn: (edges: Edge[]) => Edge[]) => void;
  project: () => Project;
  load: (project: Project, fit?: boolean) => void;
  loads: number;
  arrange: () => void;
  addBlock: (typeId: string, position: { x: number; y: number }) => string | null;
  setParam: (nodeId: string, name: string, value: unknown) => void;
  commit: () => void;
  undo: () => void;
  redo: () => void;
  validation: Validation | null;
  dataIssues: { severity: string; message: string; node_id: string }[];
  selected: string | null;
  setSelected: (id: string | null) => void;
  selectNode: (id: string) => void;
  toast: Toast;
  notify: (text: string, error?: boolean) => void;
  theme: string;
  toggleTheme: () => void;
  version: string;
}

const Ctx = createContext<Store | null>(null);

export function useStore(): Store {
  const store = useContext(Ctx);
  if (!store) throw new Error("useStore outside StoreProvider");
  return store;
}

const AUTOSAVE = "aime.autosave";

export function StoreProvider({ children }: { children: ReactNode }) {
  const [blocks, setBlocks] = useState<BlockDef[]>([]);
  const [name, setName] = useState("untitled");
  const [nodes, setNodes] = useState<FlowNode[]>([]);
  const [edges, setEdgesState] = useState<Edge[]>([]);
  const [meta, setMeta] = useState<Record<string, unknown>>({});
  const [schema, setSchema] = useState(2);
  const [loads, setLoads] = useState(0);
  const [validation, setValidation] = useState<Validation | null>(null);
  const [dataIssues, setDataIssues] = useState<Store["dataIssues"]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [toast, setToast] = useState<Toast>(null);
  const [theme, setTheme] = useState(localStorage.getItem("aime.theme") ?? "dark");
  const [version, setVersion] = useState("");
  const past = useRef<Project[]>([]);
  const future = useRef<Project[]>([]);
  const toastTimer = useRef<number>(0);

  const blockMap = useMemo(() => new Map(blocks.map((b) => [b.type_id, b])), [blocks]);

  const notify = useCallback((text: string, error = false) => {
    setToast({ text, error });
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), error ? 7000 : 3500);
  }, []);

  const project = useCallback(() => toProject(name, nodes, edges, meta, schema),
                              [name, nodes, edges, meta, schema]);

  const load = useCallback((p: Project, fit = true) => {
    const { nodes: n, edges: e } = fromProject(p);
    setNodes(hasOverlaps(n) ? autoLayout(n, e) : n);
    setEdgesState(e);
    setName(p.name || "untitled");
    setMeta((p.meta as Record<string, unknown>) ?? {});
    setSchema(Number(p.schema_version ?? 2));
    setSelected(null);
    if (fit) setLoads((c) => c + 1);
  }, []);

  // ---- history: snapshot before structural edits
  const latest = useRef(project);
  latest.current = project;
  const commit = useCallback(() => {
    past.current.push(latest.current());
    if (past.current.length > 100) past.current.shift();
    future.current = [];
  }, []);
  const undo = useCallback(() => {
    const prev = past.current.pop();
    if (!prev) return;
    future.current.push(latest.current());
    load(prev, false);
  }, [load]);
  const redo = useCallback(() => {
    const next = future.current.pop();
    if (!next) return;
    past.current.push(latest.current());
    load(next, false);
  }, [load]);

  const onNodesChange = useCallback((changes: NodeChange<FlowNode>[]) => {
    if (changes.some((c) => c.type === "remove")) commit();
    setNodes((ns) => applyNodeChanges(changes, ns));
  }, [commit]);
  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    if (changes.some((c) => c.type === "remove")) commit();
    setEdgesState((es) => applyEdgeChanges(changes, es));
  }, [commit]);
  const setEdges = useCallback((fn: (edges: Edge[]) => Edge[]) => setEdgesState(fn), []);

  const addBlock = useCallback((typeId: string, position: { x: number; y: number }) => {
    const def = blockMap.get(typeId);
    if (!def) return null;
    commit();
    let id = "";
    setNodes((ns) => {
      id = newNodeId(typeId, new Set(ns.map((n) => n.id)));
      return [...ns.map((n) => ({ ...n, selected: false })),
              { id, type: "block", position, selected: true,
                data: { typeId, params: defaultParams(def) } }];
    });
    return id;
  }, [blockMap, commit]);

  const setParam = useCallback((nodeId: string, param: string, value: unknown) => {
    setNodes((ns) => ns.map((n) => (n.id === nodeId
      ? { ...n, data: { ...n.data, params: { ...n.data.params, [param]: value } } } : n)));
  }, []);

  // ---- boot: catalog, version, autosaved project or the demo sample
  useEffect(() => {
    let attempt = 0;
    const fetchBlocks = () => api.blocks().then((r) => setBlocks(r.blocks)).catch((e) => {
      if (++attempt < 6) window.setTimeout(fetchBlocks, 1000 * attempt); // server starting up
      else notify(`Could not load the block catalog: ${e.message ?? e}`, true);
    });
    fetchBlocks();
    api.info().then((r) => {
      setVersion(r.version);
      if (r.auth && !localStorage.getItem("aime.token")) {
        const token = window.prompt("This server needs an access token:");
        if (token) {
          api.setToken(token);
          window.location.reload();
        }
      }
    }).catch(() => undefined);
    const saved = localStorage.getItem(AUTOSAVE);
    if (saved) {
      try {
        load(JSON.parse(saved));
        return;
      } catch {
        /* fall through to the sample */
      }
    }
    api.sample("mnist_cnn.json").then(load).catch(() => undefined);
  }, [load, notify]);

  // ---- live validation (debounced, structure only)
  const key = structureKey(nodes, edges);
  useEffect(() => {
    if (!nodes.length) {
      setValidation(null);
      setDataIssues([]);
      return;
    }
    const graph = latest.current();
    const timer = window.setTimeout(() => {
      api.validate(graph).then(setValidation).catch(() => setValidation(null));
      if (graph.nodes.some((n) => n.type.startsWith("data."))) {
        api.dataIssues(graph).then((r) => setDataIssues(r.issues)).catch(() => setDataIssues([]));
      } else {
        setDataIssues([]);
      }
    }, 350);
    return () => window.clearTimeout(timer);
  }, [key, nodes.length]);

  // ---- autosave (positions included)
  useEffect(() => {
    if (!nodes.length) return;
    const timer = window.setTimeout(
      () => localStorage.setItem(AUTOSAVE, JSON.stringify(latest.current())), 800);
    return () => window.clearTimeout(timer);
  }, [nodes, edges, name]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("aime.theme", theme);
  }, [theme]);
  const toggleTheme = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);

  const selectNode = useCallback((id: string) => {
    setNodes((ns) => ns.map((n) => ({ ...n, selected: n.id === id })));
    setSelected(id);
  }, []);

  const arrange = useCallback(() => {
    commit();
    setNodes((ns) => autoLayout(ns, latestEdges.current));
    setLoads((c) => c + 1);
  }, [commit]);
  const latestEdges = useRef<Edge[]>([]);
  latestEdges.current = edges;

  const store: Store = {
    blocks, blockMap, name, setName, nodes, edges, onNodesChange, onEdgesChange, setEdges,
    project, load, loads, arrange, addBlock, setParam, commit, undo, redo, validation, dataIssues, selected,
    setSelected, selectNode, toast, notify, theme, toggleTheme, version,
  };
  return <Ctx.Provider value={store}>{children}</Ctx.Provider>;
}
