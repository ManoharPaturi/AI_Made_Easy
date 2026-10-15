// Recently opened or saved projects (3.3), newest first, kept in this browser.
const KEY = "aime.recent";
const MAX = 8;

export function recentProjects(): string[] {
  try {
    const list = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(list) ? list.filter((x) => typeof x === "string").slice(0, MAX) : [];
  } catch {
    return [];
  }
}

export function rememberProject(name: string): void {
  if (!name) return;
  const list = [name, ...recentProjects().filter((n) => n !== name)].slice(0, MAX);
  localStorage.setItem(KEY, JSON.stringify(list));
}

export function forgetProject(name: string): void {
  localStorage.setItem(KEY, JSON.stringify(recentProjects().filter((n) => n !== name)));
}

/** Sort key: recent projects first, in recency order. */
export function rank(recent: string[], name: string): number {
  const i = recent.indexOf(name);
  return i < 0 ? MAX : i;
}
