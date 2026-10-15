import { Handle, type NodeProps, Position } from "@xyflow/react";
import { memo, useState } from "react";

import { type FlowNode, paramSummary, shapeLabel } from "../project";
import { useStore } from "../state";

function BlockNodeView({ id, data, selected }: NodeProps<FlowNode>) {
  const { blockMap, validation, dataIssues } = useStore();
  const def = blockMap.get(data.typeId);
  const issues = validation?.issues.filter((i) => i.node === id) ?? [];
  const dataWarn = dataIssues.some((i) => i.node_id === id);
  const severity = issues.some((i) => i.severity === "error") ? "error"
    : issues.length || dataWarn ? "warning" : "";
  const shape = validation?.shapes[id];
  const inputs = def?.inputs ?? [];
  const outputs = def?.outputs ?? [];
  const title = issues.map((i) => i.message).join("\n");
  const design = validation?.designs?.[id];
  const [open, setOpen] = useState(false);
  // ports spread over the card; an expanded stage keeps them beside its header
  const place = (i: number, n: number) => open ? `${22 + i * 14}px` : `${((i + 1) / (n + 1)) * 100}%`;
  return (
    <div className={`block-node ${selected ? "selected" : ""} ${severity ? `has-${severity}` : ""}`}
         title={title || def?.description} data-testid={`node-${id}`}>
      <div className="head" style={{ background: def?.color ?? "var(--text-muted)" }}>
        <span>{def?.display_name ?? data.typeId}</span>
        {severity && <span className={`sev-${severity}`}>{severity === "error" ? "✕" : "⚠"}</span>}
      </div>
      <div className="body">
        {shape && <span className="shape">{shapeLabel(shape)} </span>}
        {paramSummary(def, data.params)}
      </div>
      {design && (
        <button className="nodrag inline-toggle" data-testid={`expand-${id}`}
                title={design.error || `${design.name}: ${design.kind}`}
                onClick={(e) => { e.stopPropagation(); setOpen(!open); }}>
          {open ? "▾" : "▸"} {design.error ? "design: not found"
            : `${design.name} · ${design.blocks.length} blocks`}
        </button>)}
      {design && open && !design.error && (
        <div className="inline-design" data-testid={`inline-${id}`}>
          <div className="inline-kind">{design.kind}</div>
          {design.blocks.map((b, i) => (
            <div key={b.id} className={`inline-block ${b.model ? "" : "config"}`}>
              {b.model && i > 0 && <span className="inline-arrow">↓</span>}
              <span className="inline-dot" style={{ background: b.color }} />
              <span className="inline-name">{b.name}</span>
              <span className="inline-summary">{b.summary}</span>
            </div>))}
        </div>)}
      {inputs.map((p, i) => (
        <Handle key={`in-${p.name}`} id={p.name} type="target" position={Position.Left}
                style={{ top: place(i, inputs.length) }}
                title={p.role && p.role !== "tensor" ? `${p.name} (${p.role})` : p.name} />
      ))}
      {outputs.map((p, i) => (
        <Handle key={`out-${p.name}`} id={p.name} type="source" position={Position.Right}
                style={{ top: place(i, outputs.length) }}
                title={p.role && p.role !== "tensor" ? `${p.name} (${p.role})` : p.name} />
      ))}
    </div>
  );
}

export const BlockNode = memo(BlockNodeView);
