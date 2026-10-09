import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByPlaceholder(/Search \d{3} blocks/)).toBeVisible();
  await expect(page.locator(".block-node").first()).toBeVisible();
});

test("loads the example design and validates it", async ({ page }) => {
  await expect(page.getByTestId("status")).toHaveText("No problems");
  await expect(page.getByTestId("design-kind")).toHaveText(
    "Neural network · Multi-class classification");
  await expect(page.locator(".block-node")).toHaveCount(18);
  await expect(page.locator(".react-flow__edge-text").first()).toContainText("[");
});

test("drag a block from the library onto the canvas", async ({ page }) => {
  const before = await page.locator(".block-node").count();
  await page.getByPlaceholder(/Search/).fill("dense");
  await page.getByTestId("lib-core.dense").dragTo(page.getByTestId("canvas"), {
    targetPosition: { x: 300, y: 80 },
  });
  await expect(page.locator(".block-node")).toHaveCount(before + 1);
  // an unconnected layer is reported by live validation
  await expect(page.getByTestId("status")).not.toHaveText("No problems");
  await expect(page.getByTestId("problems")).toContainText(/Dense|dense|not connected|input/i);
  // the new block is selected: edit a parameter in the inspector
  await page.locator("#param-units").fill("42");
  await page.locator("#param-units").press("Enter");
  await expect(page.locator(".block-node.selected")).toContainText("units 42");
});

test("code preview and pages", async ({ page }) => {
  await page.getByRole("button", { name: "Code" }).click();
  await expect(page.getByTestId("code-view")).toContainText("class");
  await page.getByRole("button", { name: "Experiments" }).click();
  await expect(page.getByTestId("experiments-page")).toBeVisible();
  await page.getByRole("button", { name: "Models" }).click();
  await expect(page.getByTestId("models-page")).toContainText("Model registry");
  await page.getByRole("button", { name: "Data" }).click();
  await expect(page.getByTestId("data-page")).toContainText("Torchvision");
});

test("open an example from the Open dialog", async ({ page }) => {
  await page.getByRole("button", { name: "Open" }).click();
  await page.getByTestId("sample-iris_mlp.json").click();
  await expect(page.getByLabel("Project name").first()).toHaveValue("iris_mlp");
  await expect(page.getByTestId("status")).toHaveText("No problems");
});

test("summary shows costs and checks the resource budget", async ({ page }) => {
  await page.getByRole("button", { name: "Summary" }).click();
  await expect(page.getByTestId("cost-tiles")).toContainText("FLOPs");
  await page.getByLabel("Target device").selectOption("raspberry_pi_5");
  await expect(page.getByTestId("memory-bar")).toContainText("/ 6 GB");
  await page.getByLabel("Max training memory").fill("0.001");
  await expect(page.getByTestId("memory-bar")).toHaveClass(/over/);
  await expect(page.getByTestId("problems")).toContainText("Training needs about");
});

test("vision sample: detection design with box-aware pipeline", async ({ page }) => {
  await page.getByRole("button", { name: "Open" }).click();
  await page.getByTestId("sample-shapes_detection.json").click();
  await expect(page.getByTestId("design-kind")).toHaveText("Neural network · Object detection");
  await expect(page.getByTestId("status")).toHaveText("No problems");
  await page.getByRole("button", { name: "Summary" }).click();
  await expect(page.getByTestId("cost-tiles")).toContainText("19.40M");
  await page.getByPlaceholder(/Search/).fill("detector");
  await expect(page.getByTestId("lib-vision.detector")).toBeVisible();
});

test("forecasting and speech samples", async ({ page }) => {
  await page.getByRole("button", { name: "Open" }).click();
  await page.getByTestId("sample-demand_forecasting.json").click();
  await expect(page.getByTestId("design-kind")).toHaveText(
    "Neural network · Time-series forecasting");
  await expect(page.getByTestId("status")).toHaveText("No problems");
  await page.getByRole("button", { name: "Open" }).click();
  await page.getByTestId("sample-speech_recognition_ctc.json").click();
  await expect(page.getByTestId("design-kind")).toHaveText(
    "Neural network · Speech recognition (CTC)");
  await expect(page.getByTestId("status")).toHaveText("No problems");
  await page.getByPlaceholder(/Search/).fill("mamba");
  await expect(page.getByTestId("lib-seq.mamba")).toBeVisible();
});

test("generative samples", async ({ page }) => {
  await page.getByRole("button", { name: "Open" }).click();
  await page.getByTestId("sample-shapes_diffusion.json").click();
  await expect(page.getByTestId("design-kind")).toHaveText(
    "Neural network · Image generation (diffusion)");
  await expect(page.getByTestId("status")).toHaveText("No problems");
  await page.getByRole("button", { name: "Open" }).click();
  await page.getByTestId("sample-tiny_gpt.json").click();
  await expect(page.getByTestId("design-kind")).toHaveText("Neural network · Language modeling");
  await expect(page.getByTestId("status")).toHaveText("No problems");
  await page.getByPlaceholder(/Search/).fill("discriminator");
  await expect(page.getByTestId("lib-gen.discriminator")).toBeVisible();
});

test("probabilistic samples and the CPD table editor", async ({ page }) => {
  await page.getByRole("button", { name: "Open" }).click();
  await page.getByTestId("sample-student_network.json").click();
  await expect(page.getByTestId("design-kind")).toHaveText(
    "Graphical model · Probabilistic inference");
  await expect(page.getByTestId("status")).toHaveText("No problems");
  await page.locator(".block-node", { hasText: "A, B, C" }).first().click();
  await page.getByTestId("edit-table").click();
  await expect(page.getByTestId("table-editor")).toContainText("P(G | D, I)");
  await expect(page.getByTestId("table-editor")).toContainText("every column sums to 1");
  await page.getByTestId("table-editor").locator("input").first().fill("0.9");
  await expect(page.getByTestId("table-editor")).toContainText("do not sum to 1");
  await page.getByRole("button", { name: "Cancel" }).click();
  for (const [sample, kind] of [
    ["hierarchical_regression.json", "Probabilistic program · Bayesian modeling"],
    ["gp_trend_seasonality.json",
     "Gaussian process · Gaussian-process regression / classification"],
    ["flow_checkerboard_spline.json", "Neural network · Density estimation (normalizing flow)"],
    ["structural_time_series.json", "State-space model · State-space forecasting"],
  ]) {
    await page.getByRole("button", { name: "Open" }).click();
    await page.getByTestId(`sample-${sample}`).click();
    await expect(page.getByTestId("design-kind")).toHaveText(kind);
    await expect(page.getByTestId("status")).toHaveText("No problems");
  }
  await page.getByPlaceholder(/Search/).fill("conformal");
  await expect(page.getByTestId("lib-eval.conformal")).toBeVisible();
});

test("graph, tabular, recommender and reinforcement-learning samples", async ({ page }) => {
  for (const [sample, kind] of [
    ["graph_node_classification.json", "Neural network · Node classification"],
    ["graph_classification_gin.json", "Neural network · Graph classification"],
    ["graph_link_prediction.json", "Neural network · Link prediction"],
    ["tabular_ft_transformer.json", "Neural network · Multi-class classification"],
    ["recommender_dlrm.json", "Neural network · Recommendation"],
    ["rl_cartpole_ppo.json", "Neural network · Reinforcement learning"],
  ]) {
    await page.getByRole("button", { name: "Open" }).click();
    await page.getByTestId(`sample-${sample}`).click();
    await expect(page.getByTestId("design-kind")).toHaveText(kind);
    // the graph and rl extras may be missing (CI's web job): "needs … not installed" warnings
    await expect(page.getByTestId("status")).toHaveText(
      /^(graph|rl)_/.test(sample) ? /^(No problems|\d+ warning\(s\))$/ : "No problems");
  }
  await page.getByPlaceholder(/Search/).fill("transformer");
  await expect(page.getByTestId("lib-tab.ft_transformer")).toBeVisible();
});
