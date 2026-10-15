import { expect, type Page, test } from "@playwright/test";

// Canvas conveniences (3.3): copy / paste / duplicate, groups, notes, dropping a data file.
test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.removeItem("aime.autosave"));
  await page.reload();
  await expect(page.locator(".block-node").first()).toBeVisible();
});

async function selectTwo(page: Page) {
  const blocks = page.locator(".block-node");
  await blocks.nth(0).click();
  await blocks.nth(1).click({ modifiers: ["ControlOrMeta"] });
  await expect(page.locator(".block-node.selected")).toHaveCount(2);
}

/** The browser's copy / paste events, with their own clipboard data. */
async function copyPaste(page: Page): Promise<string> {
  return page.evaluate(() => {
    const out = new DataTransfer();
    window.dispatchEvent(new ClipboardEvent("copy", { clipboardData: out }));
    const text = out.getData("text/plain");
    const into = new DataTransfer();
    into.setData("text/plain", text);
    window.dispatchEvent(new ClipboardEvent("paste", { clipboardData: into }));
    return text;
  });
}

test("copy, paste and duplicate blocks with their wires", async ({ page }) => {
  const before = await page.locator(".block-node").count();
  const edgesBefore = await page.locator(".react-flow__edge").count();
  await selectTwo(page);
  const text = await copyPaste(page);
  expect(JSON.parse(text).aime_clipboard).toBe(1);
  await expect(page.locator(".block-node")).toHaveCount(before + 2);
  await expect(page.locator(".block-node.selected")).toHaveCount(2);   // the pasted pair
  expect(await page.locator(".react-flow__edge").count()).toBeGreaterThanOrEqual(edgesBefore);
  await page.keyboard.press("ControlOrMeta+d");
  await expect(page.locator(".block-node")).toHaveCount(before + 4);
  await page.keyboard.press("ControlOrMeta+z");
  await expect(page.locator(".block-node")).toHaveCount(before + 2);
  // pasting text that is not a design does nothing
  const after = await page.evaluate(() => {
    const dt = new DataTransfer();
    dt.setData("text/plain", "hello");
    window.dispatchEvent(new ClipboardEvent("paste", { clipboardData: dt }));
    return document.querySelectorAll(".block-node").length;
  });
  expect(after).toBe(before + 2);
});

test("group blocks, rename the group, ungroup", async ({ page }) => {
  await selectTwo(page);
  await page.getByTestId("group").click();
  const frame = page.getByTestId("group-group1");
  await expect(frame).toBeVisible();
  await expect(frame).toContainText("Group · 2");
  await frame.getByRole("button", { name: "Rename Group" }).click();
  await page.getByLabel("Group name").fill("Feature extractor");
  await page.getByLabel("Group name").press("Enter");
  await expect(frame).toContainText("Feature extractor");
  const saved = await page.evaluate(async () => {
    await new Promise((r) => setTimeout(r, 1000));
    return JSON.parse(localStorage.getItem("aime.autosave") ?? "{}").meta.groups;
  });
  expect(saved[0].title).toBe("Feature extractor");
  expect(saved[0].nodes).toHaveLength(2);
  // clicking the title selects the group's blocks
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
  await frame.locator(".group-title").click();
  await expect(page.locator(".block-node.selected")).toHaveCount(2);
  await frame.getByRole("button", { name: "Ungroup Feature extractor" }).click();
  await expect(frame).toHaveCount(0);
});

test("sticky notes are saved with the design", async ({ page }) => {
  await page.getByTestId("add-note").click();
  await page.getByTestId("note-text-note1").fill("Try a wider first layer");
  await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem("aime.autosave")
    ?? "{}").meta?.notes?.[0]?.text)).toBe("Try a wider first layer");
  await page.reload();
  await expect(page.getByTestId("note-text-note1")).toHaveValue("Try a wider first layer");
  // notes never reach the code
  await page.getByRole("button", { name: "Code" }).click();
  await expect(page.getByTestId("code-view")).not.toContainText("wider first layer");
  await page.getByTestId("note-note1").getByRole("button", { name: "Delete note" }).click();
  await expect(page.getByTestId("note-note1")).toHaveCount(0);
});

test("notes stay visible when the block catalog arrives late", async ({ page }) => {
  await page.getByTestId("add-note").click();
  await page.getByTestId("note-text-note1").fill("hello");
  await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem("aime.autosave")
    ?? "{}").meta?.notes?.[0]?.text)).toBe("hello");
  // a busy server: the catalog comes after the restored design is drawn
  await page.route(/\/api\/(blocks|info)$/, async (route) => {
    await new Promise((r) => setTimeout(r, 2000));
    await route.continue();
  });
  await page.reload();
  await expect(page.getByTestId("note-note1")).toBeVisible();
  await expect(page.locator(".block-item").first()).toBeVisible({ timeout: 15_000 });
  await page.waitForTimeout(500);
  await expect(page.getByTestId("note-note1")).toBeVisible();
});

test("dropping a CSV on the canvas starts a project from it", async ({ page }) => {
  const rows = ["age,income,churn", ...Array.from({ length: 60 }, (_, i) =>
    `${20 + (i % 40)},${1000 + i * 37},${i % 3 === 0 ? "yes" : "no"}`)].join("\n");
  const dt = await page.evaluateHandle((csv) => {
    const transfer = new DataTransfer();
    transfer.items.add(new File([csv], "customers.csv", { type: "text/csv" }));
    return transfer;
  }, rows);
  await page.dispatchEvent("[data-testid=canvas]", "drop", { dataTransfer: dt });
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByRole("dialog")).toContainText(/customers|60 rows|churn/i);
  // anything else is refused with a message
  const bad = await page.evaluateHandle(() => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(["x"], "virus.exe"));
    return transfer;
  });
  await page.getByRole("dialog").getByRole("button", { name: /Close|Cancel/ }).first().click();
  await page.dispatchEvent("[data-testid=canvas]", "drop", { dataTransfer: bad });
  await expect(page.getByRole("status")).toContainText("virus.exe");
});

test("a run's feature importance and what changed since it", async ({ page, request }) => {
  const chain = ["in", "d1", "act", "d2", "out"];
  const graph = {
    name: "e2e_explain",
    nodes: [
      { id: "in", type: "core.input", params: { shape: "6" }, position: [0, 0] },
      { id: "d1", type: "core.dense", params: { units: 8 }, position: [0, 0] },
      { id: "act", type: "core.relu", params: {}, position: [0, 0] },
      { id: "d2", type: "core.dense", params: { units: 3 }, position: [0, 0] },
      { id: "out", type: "core.output", params: {}, position: [0, 0] },
      { id: "data", type: "data.synthetic", position: [0, 0],
        params: { kind: "classification", n_samples: 150, n_features: 6, n_classes: 3, seed: 2 } },
      { id: "opt", type: "train.adam", params: { lr: 0.01 }, position: [0, 0] },
      { id: "trainer", type: "train.trainer", params: { epochs: 2, batch_size: 32, device: "cpu" },
        position: [0, 0] },
      { id: "why", type: "eval.permutation_importance", params: { n_repeats: 2 }, position: [0, 0] },
    ],
    edges: chain.slice(1).map((b, i) => ({ from: `${chain[i]}/out`, to: `${b}/in` })),
  };
  const started = await (await request.post("/api/runs", { data: { graph, project: "e2e_explain" } })).json();
  await expect.poll(async () => (await (await request.get(`/api/runs/${started.run_id}`)).json()).status,
    { timeout: 120_000 }).toBe("finished");
  await page.getByRole("button", { name: "Experiments", exact: true }).click();
  await page.getByLabel(/only “/).uncheck();   // every project's runs
  await page.getByRole("cell", { name: started.run_id }).click();
  await expect(page.getByTestId("run-explain")).toContainText("feature 0");
  await page.getByTestId("diff-run").click();
  await expect(page.getByTestId("design-diff")).toContainText(/block|wire|change|same/);
});
