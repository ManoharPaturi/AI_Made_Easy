// Sticky notes and group frames on the canvas (3.3). Both live in the design's meta
// (canvas.ts); code generation ignores them.
import { Handle, type NodeProps, NodeResizer, Position } from "@xyflow/react";
import { memo, useState } from "react";

import type { Group, Note } from "../canvas";
import { useStore } from "../state";

type NoteData = { note: Note };
type GroupData = { group: Group };

function NoteView({ data, selected }: NodeProps & { data: NoteData }) {
  const { updateNote, removeNote } = useStore();
  const { note } = data;
  return (
    <div className={`canvas-note ${selected ? "selected" : ""}`} data-testid={`note-${note.id}`}>
      <NodeResizer isVisible={!!selected} minWidth={140} minHeight={70}
        onResizeEnd={(_e, p) => updateNote(note.id, { size: [Math.round(p.width), Math.round(p.height)],
                                                       position: [Math.round(p.x), Math.round(p.y)] })} />
      <div className="note-head">
        <span>Note</span>
        <button className="nodrag icon" aria-label="Delete note" title="Delete note"
                onClick={() => removeNote(note.id)}>×</button>
      </div>
      <textarea className="nodrag nowheel" value={note.text} placeholder="Write a note…"
                aria-label="Note text" data-testid={`note-text-${note.id}`}
                onChange={(e) => updateNote(note.id, { text: e.target.value })} />
    </div>
  );
}

function GroupView({ data }: NodeProps & { data: GroupData }) {
  const { renameGroup, ungroup, selectIds, collapseGroup } = useStore();
  const { group } = data;
  const [editing, setEditing] = useState(false);
  const folded = !!group.collapsed;
  return (
    <div className={`canvas-group${folded ? " collapsed" : ""}`}
         style={{ borderColor: group.color, background: `${group.color}${folded ? "2b" : "14"}` }}
         data-testid={`group-${group.id}`}>
      {folded && <Handle type="target" position={Position.Left} isConnectable={false} />}
      {folded && <Handle type="source" position={Position.Right} isConnectable={false} />}
      <div className="group-head" style={{ color: group.color }}>
        <button className="nodrag icon" aria-expanded={!folded}
                aria-label={`${folded ? "Expand" : "Collapse"} ${group.title}`}
                title={folded ? "Show the group's blocks" : "Fold the group into one card"}
                onClick={() => collapseGroup(group.id, !folded)}>{folded ? "▸" : "▾"}</button>
        {editing
          ? <input className="nodrag" autoFocus defaultValue={group.title} aria-label="Group name"
                   onBlur={(e) => { renameGroup(group.id, e.target.value.trim() || "Group"); setEditing(false); }}
                   onKeyDown={(e) => { if (e.key === "Enter" || e.key === "Escape") (e.target as HTMLInputElement).blur(); }} />
          : <>
              <button className="nodrag group-title"
                      title={folded ? "Expand to edit its blocks" : "Select the group's blocks"}
                      onClick={() => (folded ? collapseGroup(group.id, false) : selectIds(group.nodes))}>
                {group.title} <span className="muted">· {group.nodes.length}</span></button>
              <button className="nodrag icon" aria-label={`Rename ${group.title}`} title="Rename"
                      onClick={() => setEditing(true)}>✎</button>
            </>}
        <button className="nodrag icon" aria-label={`Ungroup ${group.title}`} title="Ungroup"
                onClick={() => ungroup(group.id)}>×</button>
      </div>
      {folded && <div className="group-summary muted">{group.nodes.length} blocks folded</div>}
    </div>
  );
}

export const NoteNode = memo(NoteView);
export const GroupNode = memo(GroupView);
