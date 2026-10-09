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
