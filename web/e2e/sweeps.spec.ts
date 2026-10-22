import { expect, test } from "@playwright/test";

// 3.7: sweeping again near the best trials, reviewed in the sweep form before it starts.
const tiny = {
  name: "sweep_again_model",
  nodes: [
    { id: "in", type: "core.input", params: { shape: "8" }, position: [0, 0] },
    { id: "d1", type: "core.dense", params: { units: 8 }, position: [0, 0] },
    { id: "act", type: "core.relu", params: {}, position: [0, 0] },
    { id: "d2", type: "core.dense", params: { units: 3 }, position: [0, 0] },
    { id: "out", type: "core.output", params: {}, position: [0, 0] },
    { id: "data", type: "data.synthetic", position: [0, 0],
      params: { kind: "classification", n_samples: 120, n_features: 8, n_classes: 3, seed: 3 } },
    { id: "opt", type: "train.adam", params: { lr: 0.01 }, position: [0, 0] },
    { id: "loss", type: "train.loss_cross_entropy", params: {}, position: [0, 0] },
    { id: "trainer", type: "train.trainer", params: { epochs: 1, batch_size: 32, device: "cpu" },
      position: [0, 0] },
  ],
  edges: [["in", "d1"], ["d1", "act"], ["act", "d2"], ["d2", "out"]]
    .map(([a, b]) => ({ from: `${a}/out`, to: `${b}/in` })),
};

test("a finished sweep can be swept again near its best trials", async ({ page, request }) => {
  test.setTimeout(240_000);
  const spec = {
    metric: "val_loss", strategy: "random", max_trials: 5, seed: 2,
    dimensions: [{ node: "opt", param: "lr", kind: "float", low: 0.0001, high: 0.5, log: true },
                 { node: "d1", param: "units", kind: "choice", values: [4, 8, 16] }],
  };
  const started = await (await request.post("/api/sweeps",
    { data: { graph: tiny, spec, project: "again" } })).json();
  await expect.poll(async () => (await (await request.get(`/api/sweeps/${started.sweep_id}`)).json()).state,
    { timeout: 200_000 }).toBe("finished");
  await page.goto("/");
  await expect(page.locator(".block-node").first()).toBeVisible();
  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByRole("button", { name: "Sweeps" }).click();
  await page.getByText(/sweep_again_model/).first().click();
  await page.getByRole("button", { name: "Sweep again near the best…" }).click();
  const changes = page.getByTestId("sweep-again-changes");
  await expect(changes).toContainText("narrowed to the best 3 of 5 trials");
  await expect(changes).toContainText("lr (opt)");
  await expect(page.getByRole("button", { name: "Start sweep" })).toBeEnabled();
  const proposal = await (await request.get(`/api/sweeps/${started.sweep_id}/again`)).json();
  const lr = proposal.spec.dimensions.find((d: { param: string }) => d.param === "lr");
  const row = page.locator("tr", { hasText: "lr" }).filter({ has: page.locator("input[type=number]") });
  await expect(row.locator("input[type=number]").first()).toHaveValue(String(lr.low));
  await expect(row.locator("input[type=checkbox]").first()).toBeChecked();
});
