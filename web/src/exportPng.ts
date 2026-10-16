// Export the canvas as a PNG (3.3) without a screenshot library: blocks, wires, group frames
// and notes are drawn onto a 2D canvas from their positions, in the current theme's colours.

export interface PngBox { id: string; x: number; y: number; w: number; h: number; title: string;
                          subtitle?: string; color: string; kind: "block" | "note" | "group" }
export interface PngWire { from: string; to: string; label?: string }

const css = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

function clip(g: CanvasRenderingContext2D, text: string, width: number): string {
  if (g.measureText(text).width <= width) return text;
  let t = text;
  while (t.length > 1 && g.measureText(`${t}…`).width > width) t = t.slice(0, -1);
  return `${t}…`;
}

/** Draw the design; returns the canvas (scale 2 for crisp text). */
export function drawDesign(boxes: PngBox[], wires: PngWire[], scale = 2): HTMLCanvasElement {
  const pad = 40;
  const xs = boxes.flatMap((b) => [b.x, b.x + b.w]);
  const ys = boxes.flatMap((b) => [b.y, b.y + b.h]);
  const x0 = Math.min(0, ...xs) - pad;
  const y0 = Math.min(0, ...ys) - pad;
  const width = Math.max(200, (boxes.length ? Math.max(...xs) : 200) - x0 + pad);
  const height = Math.max(120, (boxes.length ? Math.max(...ys) : 120) - y0 + pad);
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(width * scale);
  canvas.height = Math.ceil(height * scale);
  const g = canvas.getContext("2d")!;
  g.scale(scale, scale);
  g.translate(-x0, -y0);
  g.fillStyle = css("--surface", "#16181c");
  g.fillRect(x0, y0, width, height);
  const font = getComputedStyle(document.body).fontFamily || "sans-serif";

  for (const b of boxes.filter((x) => x.kind === "group")) {
    g.setLineDash([6, 4]);
    g.strokeStyle = b.color;
    g.lineWidth = 1.5;
    roundRect(g, b.x, b.y, b.w, b.h, 10);
    g.stroke();
    g.setLineDash([]);
    g.fillStyle = b.color;
    g.font = `600 12px ${font}`;
    g.fillText(b.title, b.x + 8, b.y + 16);
  }

  const byId = new Map(boxes.map((b) => [b.id, b]));
  g.strokeStyle = css("--wire", "#5d6470");
  g.lineWidth = 1.6;
  for (const w of wires) {
    const a = byId.get(w.from);
    const b = byId.get(w.to);
    if (!a || !b) continue;
    const sx = a.x + a.w;
    const sy = a.y + a.h / 2;
    const tx = b.x;
    const ty = b.y + b.h / 2;
    const dx = Math.max(40, Math.abs(tx - sx) / 2);
    g.beginPath();
    g.moveTo(sx, sy);
    g.bezierCurveTo(sx + dx, sy, tx - dx, ty, tx, ty);
    g.stroke();
    if (w.label) {
      g.fillStyle = css("--text-dim", "#a0a6af");
      g.font = `10px ${font}`;
      g.fillText(w.label, (sx + tx) / 2 - g.measureText(w.label).width / 2, (sy + ty) / 2 - 4);
    }
  }

  for (const b of boxes.filter((x) => x.kind === "block")) {
    g.fillStyle = css("--node-body", "#23262d");
    roundRect(g, b.x, b.y, b.w, b.h, 7);
    g.fill();
    g.strokeStyle = css("--node-border", "#3a3f48");
    g.lineWidth = 1;
    g.stroke();
    g.save();
    roundRect(g, b.x, b.y, b.w, Math.min(22, b.h), 7);
    g.clip();
    g.fillStyle = b.color;
    g.fillRect(b.x, b.y, b.w, 22);
    g.restore();
    g.fillStyle = "#111";
    g.font = `600 11px ${font}`;
    g.fillText(clip(g, b.title, b.w - 16), b.x + 10, b.y + 15);
    if (b.subtitle) {
      g.fillStyle = css("--node-subtext", "#9aa0a9");
      g.font = `11px ${font}`;
      g.fillText(clip(g, b.subtitle, b.w - 16), b.x + 10, b.y + 38);
    }
  }

  for (const b of boxes.filter((x) => x.kind === "note")) {
    g.fillStyle = "#fff3bf";
    roundRect(g, b.x, b.y, b.w, b.h, 6);
    g.fill();
    g.fillStyle = "#2b2a1f";
    g.font = `12px ${font}`;
    const words = b.title.split(/\s+/);
    let line = "";
    let y = b.y + 22;
    for (const word of words) {
      const next = line ? `${line} ${word}` : word;
      if (g.measureText(next).width > b.w - 16 && line) {
        g.fillText(line, b.x + 8, y);
        line = word;
        y += 16;
        if (y > b.y + b.h - 6) break;
      } else {
        line = next;
      }
    }
    if (line && y <= b.y + b.h - 6) g.fillText(line, b.x + 8, y);
  }
  return canvas;
}

export function downloadCanvas(canvas: HTMLCanvasElement, filename: string): void {
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = filename;
  a.click();
}
