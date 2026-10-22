import { expect, test } from "@playwright/test";

// Try it (3.5): a finished run answers a form built from what it takes, on this machine.
const tiny = {
  name: "e2e_try",
  nodes: [
    { id: "in", type: "core.input", params: { shape: "16" }, position: [0, 0] },
    { id: "d1", type: "core.dense", params: { units: 8 }, position: [0, 0] },
    { id: "act", type: "core.relu", params: {}, position: [0, 0] },
    { id: "d2", type: "core.dense", params: { units: 3 }, position: [0, 0] },
    { id: "out", type: "core.output", params: {}, position: [0, 0] },
    { id: "data", type: "data.synthetic", position: [0, 0],
      params: { kind: "classification", n_samples: 120, n_features: 16, n_classes: 3, seed: 7 } },
    { id: "opt", type: "train.adam", params: { lr: 0.01 }, position: [0, 0] },
    { id: "loss", type: "train.loss_cross_entropy", params: {}, position: [0, 0] },
    { id: "trainer", type: "train.trainer", params: { epochs: 1, batch_size: 32, device: "cpu" },
      position: [0, 0] },
  ],
  edges: [["in", "d1"], ["d1", "act"], ["act", "d2"], ["d2", "out"]]
    .map(([a, b]) => ({ from: `${a}/out`, to: `${b}/in` })),
};

test("try it: a finished run predicts from a form, and says what's wrong", async ({ page, request }) => {
  const started = await (await request.post("/api/runs", { data: { graph: tiny, project: "e2e_try" } })).json();
  const runId: string = started.run_id;
  await expect.poll(async () => (await (await request.get(`/api/runs/${runId}`)).json()).status,
    { timeout: 120_000 }).toBe("finished");

  await page.goto("/");
  await expect(page.locator(".block-node").first()).toBeVisible();
  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByLabel(/only/).uncheck();
  await page.getByTestId("runs").locator("tr", { hasText: runId }).click();
  await page.getByTestId("try-run").click();
  const dialog = page.getByRole("dialog", { name: `Try run ${runId}` });
  await expect(dialog.getByTestId("try-form")).toHaveAttribute("data-kind", "arrays", { timeout: 60_000 });
  await dialog.getByTestId("try-json").fill(JSON.stringify([Array(16).fill(0.5)]));
  await dialog.getByTestId("try-predict").click();
  const result = dialog.getByTestId("try-result");
  await expect(result).toBeVisible({ timeout: 60_000 });
  await expect(result.locator(".try-bar")).toHaveCount(3);
  await expect(result).toContainText("% sure");

  await dialog.getByTestId("try-json").fill("[[1, 2, 3]]");     // the wrong size
  await dialog.getByTestId("try-predict").click();
  await expect(dialog.getByRole("alert")).toContainText("expects");

  // 3.6: a whole file of inputs comes back as a CSV of predictions
  await expect(dialog.getByTestId("try-explain")).toHaveCount(0);   // no convolution here
  const rows = Array.from({ length: 3 }, () => Array(16).fill(0.25).join(",")).join("\n");
  await dialog.getByTestId("try-batch").setInputFiles(
    { name: "rows.csv", mimeType: "text/csv", buffer: Buffer.from(rows) });
  const batch = dialog.getByTestId("try-batch-result");
  await expect(batch).toContainText("3 predictions", { timeout: 60_000 });
  await expect(batch.getByRole("link")).toHaveAttribute("download", "rows_predictions.csv");
  await dialog.getByRole("button", { name: "Close" }).click();
  await request.delete(`/api/runs/${runId}`);
});
