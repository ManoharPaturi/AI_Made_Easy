import { expect, test } from "@playwright/test";

// Train anywhere: add a backend, test it, pick it in "Run on" and start a run there.
// A Colab hand-off (manual: no Drive) needs no account, so the whole path runs in CI.
test("backends: add, test, run on, and the run records where it trained", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.locator(".block-node").first()).toBeVisible();
  await page.getByRole("button", { name: "Training", exact: true }).click();
  const runOn = page.getByTestId("run-on");
  await expect(runOn).toHaveValue("local");

  await runOn.selectOption("__manage");
  const dialog = page.getByRole("dialog", { name: "Backends: where runs train" });
  await expect(dialog.getByTestId("backends")).toContainText("container");
  await dialog.getByLabel("Name").fill("e2e-colab");
  await dialog.getByLabel("Kind").selectOption("colab");
  await dialog.getByLabel("manual").fill("true");
  await dialog.getByTestId("save-backend").click();
  const row = dialog.locator("tr", { hasText: "e2e-colab" });
  await expect(row).toContainText("manual=true");
  await row.getByRole("button", { name: "Test" }).click();
  await expect(dialog.getByTestId("backend-test-e2e-colab")).toContainText("notebook");
  // no field takes a secret: unknown options are refused
  const bad = await request.post("/api/backends", {
    data: { name: "x", kind: "ssh", options: { host: "h", password: "p" } } });
  expect(bad.status()).toBe(400);
  await dialog.getByRole("button", { name: "Close" }).click();

  await page.getByTestId("run-on").selectOption("e2e-colab");
  await expect(page.getByTestId("run-on-note")).toContainText("reports back");
  await page.getByTestId("train").click();
  await expect(page.getByTestId("train-log")).toContainText("notebook saved", { timeout: 30_000 });

  const runs = await (await request.get("/api/runs")).json();
  const run = runs.runs.find((r: { backend?: string }) => r.backend === "e2e-colab");
  expect(run, JSON.stringify(runs.runs.map((r: { backend?: string }) => r.backend))).toBeTruthy();
  await page.getByRole("button", { name: "Experiments" }).click();
  await expect(page.getByTestId("runs")).toContainText("e2e-colab");
  await request.post(`/api/runs/${run.run_id}/stop`);
  await request.delete("/api/backends/e2e-colab");
  await page.evaluate(() => localStorage.removeItem("aime.runOn"));
});
