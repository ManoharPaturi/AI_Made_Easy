import { expect, test } from "@playwright/test";

// Export everywhere: a run whose project targets a Raspberry Pi gets the Pi's formats
// suggested (and pre-selected) in Deploy; a detector design can't use ExecuTorch and the
// reason says why.
const tiny = {
  name: "e2e_export",
  meta: { budget: { device: "raspberry_pi_5" } },
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
    { id: "trainer", type: "train.trainer", params: { epochs: 2, batch_size: 32, device: "cpu" },
      position: [0, 0] },
  ],
  edges: [["in", "d1"], ["d1", "act"], ["act", "d2"], ["d2", "out"]]
    .map(([a, b]) => ({ from: `${a}/out`, to: `${b}/in` })),
};

test("export: device-suggested formats in Deploy, and why a design can't use one", async ({ page, request }) => {
  const started = await (await request.post("/api/runs", { data: { graph: tiny, project: "e2e_export" } })).json();
  const runId: string = started.run_id;
  await expect.poll(async () => (await (await request.get(`/api/runs/${runId}`)).json()).status,
    { timeout: 120_000 }).toBe("finished");

  const formats = (await (await request.get(`/api/runs/${runId}/formats`)).json()).formats;
  const byId = Object.fromEntries(formats.map((f: { id: string }) => [f.id, f]));
  expect(byId.executorch_int8.suggested).toBe(true);
  expect(byId.onnx.suggested).toBe(false);

  await page.goto("/");
  await expect(page.locator(".block-node").first()).toBeVisible();
  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByLabel(/only/).uncheck();
  await page.getByTestId("runs").locator("tr", { hasText: runId }).click();
  await page.getByRole("button", { name: "Deploy…" }).click();
  const dialog = page.getByRole("dialog", { name: `Deploy run ${runId}` });
  const suggested = dialog.locator('[data-format="executorch_int8"]');
  await expect(suggested).toContainText("suggested for Raspberry Pi 5");
  await expect(suggested.locator("input")).toBeChecked();
  await expect(dialog.locator('[data-format="onnx"] input')).not.toBeChecked();
  await dialog.getByRole("button", { name: "Close" }).click();

  const detector = (await (await request.get("/api/samples/shapes_detection.json")).json());
  const graph = detector.graph ?? detector;
  const rows = (await (await request.post("/api/export/formats", { data: { graph } })).json()).formats;
  const et = rows.find((r: { id: string }) => r.id === "executorch");
  expect(et.available).toBe(false);
  expect(et.reason).toContain("NMS");
  await request.delete(`/api/runs/${runId}`);
});
