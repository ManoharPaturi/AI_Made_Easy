import { Fragment, useMemo } from "react";

// A small Python highlighter: keywords, strings, comments, numbers, called names.
const TOKEN = new RegExp(
  [
    String.raw`(#[^\n]*)`,                                            // comment
    String.raw`("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')`, // string
    String.raw`\b(def|class|return|import|from|as|if|elif|else|for|while|in|not|and|or|is|None|True|False|with|try|except|finally|raise|lambda|yield|pass|break|continue|global|assert|async|await)\b`,
    String.raw`\b(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\b`,                    // number
    String.raw`\b([A-Za-z_]\w*)(?=\()`,                              // call
  ].join("|"),
  "g",
);
const CLASSES = ["com", "str", "kw", "num", "fn"];

export function CodeView({ code }: { code: string }) {
  const parts = useMemo(() => {
    const out: (string | { cls: string; text: string })[] = [];
    let last = 0;
    for (const m of code.matchAll(TOKEN)) {
      if (m.index! > last) out.push(code.slice(last, m.index));
      const group = m.slice(1).findIndex((g) => g !== undefined);
      out.push({ cls: CLASSES[group], text: m[0] });
      last = m.index! + m[0].length;
    }
    out.push(code.slice(last));
    return out;
  }, [code]);
  return (
    <pre className="code" data-testid="code-view">
      {parts.map((p, i) => (typeof p === "string"
        ? <Fragment key={i}>{p}</Fragment>
        : <span key={i} className={p.cls}>{p.text}</span>))}
    </pre>
  );
}
