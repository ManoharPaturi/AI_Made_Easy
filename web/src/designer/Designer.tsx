import {
  Background, type Connection, Controls, type Edge, MiniMap, Panel, ReactFlow, ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { type DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, streamRun } from "../api";
import { LineChart, metricSeries } from "../components/LineChart";
import { edgeId, shapeLabel } from "../project";
import { useStore } from "../state";
import type { RunEvent } from "../types";
import { BlockNode } from "./BlockNode";
import { Inspector } from "./Inspector";
import { DRAG_TYPE, Library } from "./Library";

const nodeTypes = { block: BlockNode };

function Canvas() {
  const { nodes, edges, onNodesChange, onEdgesChange, setEdges, addBlock, setSelected, commit,
          validation, undo, redo, loads, arrange } = useStore();
  const flow = useReactFlow();

  // fit the view whenever a project is opened or re-arranged
  useEffect(() => {
    if (!loads) return;
    const timer = window.setTimeout(() => flow.fitView({ padding: 0.15, maxZoom: 1.2 }), 60);
    return () => window.clearTimeout(timer);
  }, [loads, flow]);
  const wrapper = useRef<HTMLDivElement>(null);

  const onConnect = useCallback((c: Connection) => {
    if (!c.source || !c.target) return;
    commit();
    const sourceHandle = c.sourceHandle ?? "out";
    const targetHandle = c.targetHandle ?? "in";
    // one wire per input port: a new connection replaces the old one
    setEdges((es) => [
      ...es.filter((e) => !(e.target === c.target && (e.targetHandle ?? "in") === targetHandle)),
      { id: edgeId(c.source, sourceHandle, c.target, targetHandle), source: c.source,
        sourceHandle, target: c.target, targetHandle },
    ]);
  }, [commit, setEdges]);

  const onDrop = useCallback((e: DragEvent) => {
    const typeId = e.dataTransfer.getData(DRAG_TYPE);
    if (!typeId) return;
    e.preventDefault();
    const id = addBlock(typeId, flow.screenToFlowPosition({ x: e.clientX, y: e.clientY }));
    if (id) setSelected(id);
  }, [addBlock, flow, setSelected]);

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      const typing = (e.target as HTMLElement)?.closest?.("input, textarea, select");
      if (!mod || typing) return;
      if (e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
      } else if (e.key.toLowerCase() === "y") {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [undo, redo]);

  // wires show the tensor shape flowing through them
  const labelled: Edge[] = useMemo(() => edges.map((e) => {
    const shape = validation?.shapes[e.source];
    return shape ? { ...e, label: shapeLabel(shape) } : e;
  }), [edges, validation]);

  return (
    <div className="canvas" ref={wrapper} onDragOver={(e) => { e.preventDefault(); }}
         onDrop={onDrop} data-testid="canvas">
      <ReactFlow
        nodes={nodes} edges={labelled} nodeTypes={nodeTypes}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect}
        onNodeDragStart={() => commit()}
        onSelectionChange={({ nodes: sel }) => setSelected(sel.length === 1 ? sel[0].id : null)}
        deleteKeyCode={["Backspace", "Delete"]} fitView minZoom={0.1} maxZoom={2.5}
        proOptions={{ hideAttribution: true }}>
        <Background gap={20} size={1} color="var(--border)" />
        <Controls showInteractive={false} />
        <Panel position="top-right">
          <button onClick={arrange} title="Lay blocks out left to right by data flow">Arrange</button>
        </Panel>
        <MiniMap pannable zoomable nodeColor={() => "var(--accent)"} maskColor="rgba(127,127,127,.18)" />
      </ReactFlow>
    </div>
  );
}

function Problems() {
  const { validation, dataIssues, project, load, notify, selectNode, nodes } = useStore();
  const issues = validation?.issues ?? [];
  const select = (id: string | null) => id && selectNode(id);
  const fix = async (index: number) => {
    try {
      const result = await api.fix(project(), index);
      load(result.graph);
      notify(`Applied: ${result.label}`);
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  if (!nodes.length) return <div className="empty">The canvas is empty.</div>;
  if (!issues.length && !dataIssues.length) {
    return <div className="sev-ok" style={{ padding: 8 }}>✓ No problems — the design is valid.</div>;
  }
  return (
    <div data-testid="problems">
      {issues.map((issue, i) => (
        <div key={`v${i}`} className="issue" onClick={() => select(issue.node)}>
          <span className={`sev-${issue.severity}`}>{issue.severity === "error" ? "✕" : "⚠"}</span>
          <span style={{ flex: 1 }}>{issue.message}
            {issue.node && <span className="muted"> — {issue.node}</span>}</span>
          {issue.fix && <button onClick={(e) => { e.stopPropagation(); fix(i); }}
                                title={issue.fix.description}>{issue.fix.label}</button>}
        </div>
      ))}
      {dataIssues.map((issue, i) => (
        <div key={`d${i}`} className="issue" onClick={() => select(issue.node_id)}>
          <span className="sev-warning">⚠</span>
          <span style={{ flex: 1 }}>{issue.message} <span className="muted">— data</span></span>
        </div>
      ))}
    </div>
  );
}

function Training() {
  const { project, name, validation, notify } = useStore();
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [epochs, setEpochs] = useState<{ epoch: number; metrics: Record<string, number> }[]>([]);
  const [log, setLog] = useState<string[]>([]);

  useEffect(() => {
    if (!runId) return;
    setEpochs([]);
    setLog([]);
    return streamRun(runId, (event: RunEvent) => {
      if (event.type === "epoch") {
        setEpochs((es) => [...es.filter((e) => e.epoch !== event.epoch),
                           { epoch: Number(event.epoch), metrics: event.metrics as Record<string, number> }]);
      } else if (event.type === "log" || event.type === "error") {
        setLog((l) => [...l.slice(-300), String(event.line ?? event.traceback ?? "")]);
      } else if (event.type === "done") {
        setStatus(event.status ? String(event.status)
          : Number(event.returncode) === 0 ? "finished" : "failed");
      } else if (event.type === "status") {
        setStatus(String(event.status));
      }
    });
  }, [runId]);

  const train = async () => {
    try {
      const { run_id } = await api.train(project(), name);
      setRunId(run_id);
      setStatus("running");
      notify(`Training started — run ${run_id}`);
    } catch (e) {
      notify(String((e as Error).message), true);
    }
  };
  const stop = async () => {
    if (runId) await api.stopRun(runId).catch(() => undefined);
  };
  const last = epochs[epochs.length - 1];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)", gap: 12,
                  height: "100%" }}>
      <div>
        <div className="toolbar" style={{ marginBottom: 6 }}>
          <button className="primary" onClick={train}
                  disabled={status === "running" || !validation?.valid}
                  data-testid="train">▶ Train</button>
          <button onClick={stop} disabled={status !== "running"}>Stop</button>
          <span className="dim">{runId ? `run ${runId} · ${status}` : "No run yet"}
            {last && ` · epoch ${last.epoch}`}</span>
        </div>
        <LineChart series={metricSeries(epochs, (k) => k.includes("loss"))} height={150} />
      </div>
      <pre className="code" style={{ overflow: "auto", maxHeight: 200 }} data-testid="train-log">
        {log.join("\n") || "Training output appears here."}
      </pre>
    </div>
  );
}

export function Designer() {
  const { addBlock, setSelected } = useStore();
  const [bottom, setBottom] = useState("problems");
  const flowRef = useRef<{ add: (t: string) => void }>({ add: () => undefined });
  return (
    <ReactFlowProvider>
      <div className="workspace">
        <Library onAdd={(t) => flowRef.current.add(t)} />
        <div className="center-col">
          <AddHelper refObj={flowRef} addBlock={addBlock} setSelected={setSelected} />
          <Canvas />
          <section className="panel bottom-panel">
            <div className="tabs">
              {["problems", "training"].map((t) => (
                <button key={t} className={bottom === t ? "active" : ""} onClick={() => setBottom(t)}>
                  {t[0].toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
            <div className="tab-body">{bottom === "problems" ? <Problems /> : <Training />}</div>
          </section>
        </div>
        <Inspector />
      </div>
    </ReactFlowProvider>
  );
}

/** Places double-clicked library blocks at the centre of the visible canvas. */
function AddHelper({ refObj, addBlock, setSelected }: {
  refObj: { current: { add: (t: string) => void } };
  addBlock: (t: string, p: { x: number; y: number }) => string | null;
  setSelected: (id: string | null) => void;
}) {
  const flow = useReactFlow();
  refObj.current.add = (typeId: string) => {
    const el = document.querySelector(".canvas")?.getBoundingClientRect();
    const center = el ? { x: el.left + el.width / 2, y: el.top + el.height / 2 } : { x: 400, y: 300 };
    const id = addBlock(typeId, flow.screenToFlowPosition(center));
    if (id) setSelected(id);
  };
  return null;
}
