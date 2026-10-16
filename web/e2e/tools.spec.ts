import { expect, test } from "@playwright/test";
import { fileMenu } from "./helpers";

// Work with your tools (3.2): tracking for the project, run history as a table, a model's
// Share dialog (formats, MLflow, the Hub with a confirmation) and the MLflow import option.
const tiny = {
  name: "e2e_tools",
  nodes: [
    { id: "in", type: "core.input", params: { shape: "8" }, position: [0, 0] },
    { id: "d1", type: "core.dense", params: { units: 3 }, position: [0, 0] },
    { id: "out", type: "core.output", params: {}, position: [0, 0] },
    { id: "data", type: "data.synthetic", position: [0, 0],
      params: { kind: "classification", n_samples: 90, n_features: 8, n_classes: 3, seed: 1 } },
    { id: "opt", type: "train.adam", params: { lr: 0.01 }, position: [0, 0] },
    { id: "loss", type: "train.loss_cross_entropy", params: {}, position: [0, 0] },
    { id: "trainer", type: "train.trainer", params: { epochs: 1, batch_size: 32, device: "cpu" },
      position: [0, 0] },
  ],
  edges: [["in", "d1"], ["d1", "out"]].map(([a, b]) => ({ from: `${a}/out`, to: `${b}/in` })),
};

test("tools: tracking choices, run history export, share and MLflow import", async ({ page, request }) => {
  const started = await (await request.post("/api/runs", { data: { graph: tiny, project: "e2e_tools" } })).json();
  const runId: string = started.run_id;
  await expect.poll(async () => (await (await request.get(`/api/runs/${runId}`)).json()).status,
    { timeout: 120_000 }).toBe("finished");
  const model = await (await request.post("/api/models", { data: { run_id: runId, name: "e2e-tools" } })).json();

  await page.goto("/");
  await expect(page.locator(".block-node").first()).toBeVisible();

  // Tools: both trackers are listed; the run history downloads as CSV
  await page.getByTestId("open-tools").click();
  const tools = page.getByRole("dialog", { name: /Tools/ });
  await expect(tools.getByTestId("tools-tracking")).toContainText("MLflow");
  await expect(tools.getByTestId("tools-tracking")).toContainText("Weights & Biases");
  const href = await tools.getByRole("link").first().getAttribute("href");
  expect(href).toContain("/api/export/runs?format=csv");
  const csv = await (await request.get(href!)).text();
  expect(csv.split("\n")[0]).toContain("run_id");
  expect(csv).toContain(runId);
  // a bad dataset source is refused with a message, nothing is downloaded
  await tools.getByTestId("dataset-source").fill("nowhere:x");
  await tools.getByTestId("dataset-fetch").click();
  await expect(page.getByRole("status")).toContainText(/kaggle|hf/i);
  await tools.getByRole("button", { name: "Close" }).click();

  // Backends opens from the header too
  await page.getByTestId("open-backends").click();
  await expect(page.getByRole("dialog", { name: "Backends: where runs train" })).toBeVisible();
  await page.getByRole("dialog").getByRole("button", { name: "Close" }).click();

  // Share: formats, and publishing asks first (declined here: nothing is sent)
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page.getByTestId(`share-e2e-tools-${model.version}`).click();
  const share = page.getByRole("dialog", { name: `Share e2e-tools v${model.version}` });
  await expect(share.locator("[data-format]").first()).toBeVisible();
  const download = share.getByTestId("share-download").locator("xpath=..");
  expect(await download.getAttribute("href")).toContain(`/api/models/e2e-tools/${model.version}/deploy`);
  const hubCalls: string[] = [];
  page.on("request", (r) => r.url().includes("/api/hub/") && hubCalls.push(r.url()));
  await share.getByTestId("share-repo").fill("someone/e2e-tools");
  await share.getByTestId("share-hub").click();
  const ask = page.getByTestId("ask-confirm");   // the in-app confirm dialog
  await expect(ask).toContainText("someone/e2e-tools");
  await expect(ask).toContainText("private");
  await ask.getByRole("button", { name: "Cancel" }).click();
  await expect(ask).toHaveCount(0);
  await expect(share).toBeVisible();   // Escape / Cancel closes only the question
  expect(hubCalls).toEqual([]);

  const status = await (await request.get("/api/tracking")).json();
  if (status.trackers.find((t: { id: string; installed: boolean }) => t.id === "mlflow")?.installed) {
    await share.getByTestId("share-mlflow").click();
    await expect(share.getByTestId("share-result")).toContainText("Registered in MLflow",
      { timeout: 30_000 });   // MLflow creates its database on first use
    await share.getByRole("button", { name: "Close" }).click();
    await expect(page.getByTestId("mirror-mlflow")).toBeVisible();
  } else {
    await share.getByRole("button", { name: "Close" }).click();
  }

  // Import model: MLflow takes a URI instead of a file
  await fileMenu(page, "Import model…");
  const imp = page.getByRole("dialog", { name: "Import model" });
  await imp.getByLabel("Format").selectOption("mlflow");
  await expect(imp.getByLabel("Model URI")).toBeVisible();
  await expect(imp.getByLabel("File")).toHaveCount(0);
  await imp.getByRole("button", { name: "Close" }).click();

  await request.delete(`/api/models/e2e-tools/${model.version}`);
});
