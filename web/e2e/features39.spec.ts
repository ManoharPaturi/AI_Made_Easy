import { expect, test } from "@playwright/test";

// 3.9: a stopped run resumes from its last saved epoch, as the same run.
const edges = (ids: string[]) => ids.slice(1).map((b, i) => ({ from: `${ids[i]}/out`, to: `${b}/in` }));

const long = {
  schema_version: 3, name: "e2e_resume", meta: {},
  nodes: [
    { id: "in", type: "core.input", params: { shape: "8" }, position: [0, 0] },
    { id: "d1", type: "core.dense", params: { units: 16 }, position: [0, 0] },
    { id: "act", type: "core.relu", params: {}, position: [0, 0] },
    { id: "d2", type: "core.dense", params: { units: 3 }, position: [0, 0] },
    { id: "out", type: "core.output", params: {}, position: [0, 0] },
    { id: "data", type: "data.synthetic", position: [0, 0],
      params: { kind: "classification", n_samples: 400, n_features: 8, n_classes: 3, seed: 3 } },
    { id: "opt", type: "train.adam", params: { lr: 0.01 }, position: [0, 0] },
    { id: "loss", type: "train.loss_cross_entropy", params: {}, position: [0, 0] },
    { id: "trainer", type: "train.trainer", position: [0, 0],
      params: { epochs: 3000, batch_size: 16, device: "cpu", early_stopping_patience: 0 } },
  ],
  edges: edges(["in", "d1", "act", "d2", "out"]),
};

test("a stopped run resumes from its last saved epoch", async ({ page, request }) => {
  test.setTimeout(240_000);
  const started = await request.post("/api/runs", { data: { graph: long, project: "e2e39" } });
  expect(started.ok(), await started.text()).toBeTruthy();
  const { run_id } = await started.json();
  try {
    await expect.poll(async () => (await (await request.get(`/api/runs/${run_id}`)).json()).epochs_done,
      { timeout: 120_000 }).toBeGreaterThanOrEqual(3);
    await request.post(`/api/runs/${run_id}/stop`);
    await expect.poll(async () => (await (await request.get(`/api/runs/${run_id}`)).json()).status,
      { timeout: 60_000 }).toBe("stopped");
    const info = await (await request.get(`/api/runs/${run_id}/resume`)).json();
    expect(info.resumable).toBe(true);

    await page.goto("/");
    await page.getByRole("button", { name: "Experiments" }).click();
    await page.getByLabel(/only “/).uncheck();          // runs of every project
    await page.getByTestId("runs").locator("tr", { hasText: run_id }).click();
    const resume = page.getByTestId("resume-run");
    await expect(resume).toBeEnabled();
    await expect(resume).toHaveAttribute("title", new RegExp(`after epoch ${info.after_epoch}`));
    await resume.click();
    await expect(page.getByText(`continues after epoch ${info.after_epoch}`)).toBeVisible();
    await expect.poll(async () => {
      const run = await (await request.get(`/api/runs/${run_id}`)).json();
      return run.status === "running" && run.epochs_done > info.after_epoch;
    }, { timeout: 60_000 }).toBe(true);
  } finally {
    await request.post(`/api/runs/${run_id}/stop`);
  }
});
