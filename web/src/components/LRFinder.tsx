// 3.8: find a learning rate. A short range test raises the rate batch by batch; the
// chart shows the smoothed loss against the rate and marks the suggested one.
import { useEffect, useState } from "react";

import { api } from "../api";
import { useStore } from "../state";
import type { LRFindResult } from "../types";
import { LineChart } from "./LineChart";
import { Modal } from "./Modal";

const FINAL = ["finished", "failed", "stopped", "interrupted"];
const sci = (v: number) => Number(v.toPrecision(2)).toExponential(1).replace("e-", "e−");

export function LRFinder({ onUse, onClose }: {
  onUse: (lr: number) => void; onClose: () => void;
}) {
  const { project } = useStore();
  const [result, setResult] = useState<LRFindResult | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let timer = 0;
    let stopped = false;
    api.findLearningRate(project()).then(({ run_id }) => {
      const poll = async () => {
        try {
          const r = await api.learningRateResult(run_id);
          if (stopped) return;
          setResult(r);
          if (!FINAL.includes(r.state)) timer = window.setTimeout(poll, 1000);
        } catch (e) {
          setError(String((e as Error).message ?? e));
        }
      };
      poll();
    }).catch((e) => setError(String((e as Error).message ?? e)));
    return () => { stopped = true; window.clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const done = result && FINAL.includes(result.state);
  const lrs = result?.lrs ?? [];
  const points = lrs.map((lr, i) => [Math.log10(lr), result!.losses![i]] as [number, number]);
  const suggested = result?.suggested ?? null;
  return (
    <Modal title="Find a learning rate" onClose={onClose} wide
           actions={suggested ? (
             <button className="primary" data-testid="use-lr" onClick={() => onUse(suggested)}>
               Use {sci(suggested)}
             </button>) : undefined}>
      <p className="dim">
        A short run on this computer trains the design while the learning rate rises from
        10⁻⁷ batch by batch, and stops when the loss blows up. The suggestion is where the
        loss falls fastest: high enough to learn quickly, below the point where training
        becomes unstable. Nothing is saved.
      </p>
      {error && <p className="error-text" role="alert">{error}</p>}
      {!error && !done && (
        <p aria-busy="true"><span className="spinner" /> Testing learning rates…
          {lrs.length ? ` ${lrs.length} steps` : ""}</p>)}
      {done && result.state !== "finished" && (
        <p className="error-text" role="alert">The test {result.state}: {result.error}</p>)}
      {points.length > 0 && (
        <LineChart series={[{ name: "smoothed loss", points }]} xLabel="learning rate"
                   xTick={(x) => `1e${x}`}
                   marker={suggested ? { x: Math.log10(suggested), label: sci(suggested) }
                     : undefined} />)}
      {done && result.state === "finished" && (
        <p data-testid="lr-summary">
          {suggested
            ? <>Suggested <strong>{sci(suggested)}</strong> ({result.reason}); the loss was
                lowest at {sci(result.min_loss_lr ?? 0)}.</>
            : <>No suggestion: {result.reason}.</>}
        </p>)}
    </Modal>
  );
}
