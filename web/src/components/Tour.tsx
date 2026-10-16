// First-run tour (3.3): three short steps (library → canvas → train) that can be closed at any
// time and are not shown again. Help ▸ "Show the tour again" (palette) brings it back.
import { useEffect, useState } from "react";

const KEY = "aime.tour";
const STEPS = [
  { target: ".library", title: "1 · Pick blocks",
    text: "The library holds every block. Drag one onto the canvas, or select it and press Enter." },
  { target: ".canvas", title: "2 · Wire them up",
    text: "Connect ports from left to right. Problems show live underneath, with one-click fixes." },
  { target: "[data-testid=train]", fallback: ".bottom-panel", title: "3 · Train",
    text: "Open the Training tab and press ▶ Train. Runs are saved on the Experiments page." },
];

export function tourDone(): boolean {
  return localStorage.getItem(KEY) === "done";
}

export function resetTour(): void {
  localStorage.removeItem(KEY);
}

export function Tour({ onClose }: { onClose: () => void }) {
  const [step, setStep] = useState(0);
  const [box, setBox] = useState<DOMRect | null>(null);
  const s = STEPS[step];
  useEffect(() => {
    const place = () => {
      const el = document.querySelector(s.target) ?? (s.fallback ? document.querySelector(s.fallback) : null);
      setBox(el ? el.getBoundingClientRect() : null);
    };
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
  }, [s]);
  const finish = () => {
    localStorage.setItem(KEY, "done");
    onClose();
  };
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") finish(); };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // the card sits beside the highlighted area, inside the window
  const width = 300;
  const left = box ? Math.min(window.innerWidth - width - 16,
                              box.right + 12 > window.innerWidth - width ? box.left + 16 : box.right + 12) : 24;
  const top = box ? Math.min(window.innerHeight - 180, Math.max(60, box.top + 16)) : 80;
  return (
    <>
      {box && <div className="tour-ring" style={{ left: box.left, top: box.top, width: box.width,
                                                  height: box.height }} />}
      <div className="tour-card" role="dialog" aria-label="Tour" data-testid="tour"
           style={{ left, top, width }}>
        <div className="tour-step muted">Step {step + 1} of {STEPS.length}</div>
        <h4>{s.title}</h4>
        <p>{s.text}</p>
        <div className="toolbar">
          <button className="ghost" onClick={finish}>Skip tour</button>
          <span className="spacer" />
          {step > 0 && <button onClick={() => setStep(step - 1)}>Back</button>}
          {step < STEPS.length - 1
            ? <button className="primary" onClick={() => setStep(step + 1)}>Next</button>
            : <button className="primary" onClick={finish}>Done</button>}
        </div>
      </div>
    </>
  );
}
