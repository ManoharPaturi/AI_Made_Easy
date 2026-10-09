import { Handle, type NodeProps, Position } from "@xyflow/react";
import { memo } from "react";

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
  return (
    <div className={`block-node ${selected ? "selected" : ""} ${severity ? `has-${severity}` : ""}`}
         title={title || def?.description} data-testid={`node-${id}`}>
      <div className="head" style={{ background: def?.color ?? "#888" }}>
        <span>{def?.display_name ?? data.typeId}</span>
        {severity && <span className={`sev-${severity}`}>{severity === "error" ? "✕" : "⚠"}</span>}
      </div>
      <div className="body">
        {shape && <span className="shape">{shapeLabel(shape)} </span>}
        {paramSummary(def, data.params)}
      </div>
      {inputs.map((p, i) => (
        <Handle key={`in-${p.name}`} id={p.name} type="target" position={Position.Left}
                style={{ top: `${((i + 1) / (inputs.length + 1)) * 100}%` }}
                title={p.role && p.role !== "tensor" ? `${p.name} (${p.role})` : p.name} />
      ))}
      {outputs.map((p, i) => (
        <Handle key={`out-${p.name}`} id={p.name} type="source" position={Position.Right}
                style={{ top: `${((i + 1) / (outputs.length + 1)) * 100}%` }}
                title={p.role && p.role !== "tensor" ? `${p.name} (${p.role})` : p.name} />
      ))}
    </div>
  );
}

export const BlockNode = memo(BlockNodeView);
