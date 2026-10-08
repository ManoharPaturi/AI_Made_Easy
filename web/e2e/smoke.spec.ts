import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByPlaceholder(/Search \d{3} blocks/)).toBeVisible();
  await expect(page.locator(".block-node").first()).toBeVisible();
});

test("loads the example design and validates it", async ({ page }) => {
  await expect(page.getByTestId("status")).toHaveText("No problems");
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
