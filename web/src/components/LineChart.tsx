// Dependency-free SVG line chart for training curves.
export interface Series {
  name: string;
  points: [number, number][];
  color?: string;
  dashed?: boolean;
}

const PALETTE = ["#4c8dff", "#3fb950", "#e3a008", "#f85149", "#a371f7", "#39c5cf",
                 "#ff8bd1", "#9aa0a9"];

function niceTicks(lo: number, hi: number, count = 4): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (lo === hi) return [lo];
  const step = 10 ** Math.floor(Math.log10((hi - lo) / count));
  const err = ((hi - lo) / count) / step;
  const nice = step * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const ticks = [];
  for (let t = Math.ceil(lo / nice) * nice; t <= hi + 1e-12; t += nice) ticks.push(t);
  return ticks;
}

const fmt = (v: number) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)
  ? v.toExponential(1) : Number(v.toPrecision(3)).toString());

export function LineChart({ series, height = 220, xLabel = "epoch" }: {
  series: Series[]; height?: number; xLabel?: string;
}) {
  const width = 640;
  const pad = { l: 46, r: 12, t: 10, b: 26 };
  const all = series.flatMap((s) => s.points).filter(([, y]) => Number.isFinite(y));
  if (!all.length) return <div className="empty">No data yet.</div>;
  const xs = all.map(([x]) => x);
  const ys = all.map(([, y]) => y);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs, Math.min(...xs) + 1)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (width - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (y - y0) / (y1 - y0)) * (height - pad.t - pad.b);
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none"
           style={{ height }}>
        {niceTicks(y0, y1).map((t) => (
          <g key={`y${t}`}>
            <line className="axis" x1={pad.l} x2={width - pad.r} y1={sy(t)} y2={sy(t)} opacity={0.5} />
            <text x={pad.l - 6} y={sy(t) + 3} textAnchor="end">{fmt(t)}</text>
          </g>
        ))}
        {niceTicks(x0, x1, 6).filter(Number.isInteger).map((t) => (
          <text key={`x${t}`} x={sx(t)} y={height - 8} textAnchor="middle">{t}</text>
        ))}
        <text x={width - pad.r} y={height - 8} textAnchor="end">{xLabel}</text>
        {series.map((s, i) => {
          const pts = s.points.filter(([, y]) => Number.isFinite(y));
          const color = s.color ?? PALETTE[i % PALETTE.length];
          return (
            <g key={s.name}>
              <polyline fill="none" stroke={color} strokeWidth={2}
                        strokeDasharray={s.dashed ? "5 4" : undefined}
                        points={pts.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />
              {pts.length === 1 && <circle cx={sx(pts[0][0])} cy={sy(pts[0][1])} r={3} fill={color} />}
            </g>
          );
        })}
      </svg>
      <div className="legend">
        {series.map((s, i) => (
          <span key={s.name}><i style={{ background: s.color ?? PALETTE[i % PALETTE.length] }} />
            {s.name}</span>
        ))}
      </div>
    </div>
  );
}

/** Epoch events -> one series per metric. */
export function metricSeries(epochs: { epoch: number; metrics: Record<string, number> }[],
                             filter?: (key: string) => boolean): Series[] {
  const keys = [...new Set(epochs.flatMap((e) => Object.keys(e.metrics ?? {})))]
    .filter((k) => (filter ? filter(k) : true));
  return keys.map((k) => ({
    name: k,
    dashed: k.startsWith("val_"),
    points: epochs.filter((e) => typeof e.metrics?.[k] === "number")
      .map((e) => [e.epoch, e.metrics[k]] as [number, number]),
  }));
}
