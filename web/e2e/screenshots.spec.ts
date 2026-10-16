import { expect, test } from "@playwright/test";
import { fileURLToPath } from "node:url";

// Docs screenshots of the 3.2 web tools: AIME_SCREENSHOTS=1 npx playwright test screenshots
const OUT = fileURLToPath(new URL("../../docs/images", import.meta.url));
test.skip(!process.env.AIME_SCREENSHOTS, "set AIME_SCREENSHOTS=1 to refresh the docs images");

const tiny = {
  name: "churn_model",
  meta: { budget: { device: "raspberry_pi_5" } },
  nodes: [
    { id: "in", type: "core.input", params: { shape: "8" }, position: [0, 0] },
    { id: "d1", type: "core.dense", params: { units: 16 }, position: [0, 0] },
    { id: "act", type: "core.relu", params: {}, position: [0, 0] },
    { id: "d2", type: "core.dense", params: { units: 3 }, position: [0, 0] },
    { id: "out", type: "core.output", params: {}, position: [0, 0] },
    { id: "data", type: "data.synthetic", position: [0, 0],
      params: { kind: "classification", n_samples: 200, n_features: 8, n_classes: 3, seed: 3 } },
    { id: "opt", type: "train.adam", params: { lr: 0.01 }, position: [0, 0] },
    { id: "loss", type: "train.loss_cross_entropy", params: {}, position: [0, 0] },
    { id: "trainer", type: "train.trainer", params: { epochs: 3, batch_size: 32, device: "cpu" },
      position: [0, 0] },
  ],
  edges: [["in", "d1"], ["d1", "act"], ["act", "d2"], ["d2", "out"]]
    .map(([a, b]) => ({ from: `${a}/out`, to: `${b}/in` })),
};

test("docs screenshots: Tools and Share", async ({ page, request }) => {
  const started = await (await request.post("/api/runs", { data: { graph: tiny, project: "churn" } })).json();
  await expect.poll(async () => (await (await request.get(`/api/runs/${started.run_id}`)).json()).status,
    { timeout: 120_000 }).toBe("finished");
  const model = await (await request.post("/api/models", { data: { run_id: started.run_id, name: "churn-model" } })).json();

  await page.goto("/");
  await expect(page.locator(".block-node").first()).toBeVisible();
  await page.getByTestId("open-tools").click();
  await page.getByTestId("dataset-source").fill("kaggle:blastchar/telco-customer-churn");
  await page.getByRole("dialog").screenshot({ path: `${OUT}/web_tools.png` });
  await page.getByRole("dialog").getByRole("button", { name: "Close" }).click();

  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page.getByTestId(`share-churn-model-${model.version}`).click();
  const share = page.getByRole("dialog", { name: /Share churn-model/ });
  await expect(share.locator("[data-format]").first()).toBeVisible();
  await share.getByTestId("share-repo").fill("your-name/churn-model");
  await share.screenshot({ path: `${OUT}/web_share.png` });
  await share.getByRole("button", { name: "Close" }).click();
  await request.delete(`/api/models/churn-model/${model.version}`);
});

test("docs screenshots: 3.3 command palette and canvas notes / groups", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.removeItem("aime.autosave"));
  await page.reload();
  const blocks = page.locator(".block-node");
  await expect(blocks.first()).toBeVisible();
  await blocks.nth(1).click();
  await blocks.nth(2).click({ modifiers: ["ControlOrMeta"] });
  await blocks.nth(3).click({ modifiers: ["ControlOrMeta"] });
  await page.getByTestId("group").click();
  await page.getByTestId("group-group1").getByRole("button", { name: "Rename Group" }).click();
  await page.getByLabel("Group name").fill("Feature extractor");
  await page.getByLabel("Group name").press("Enter");
  await page.getByTestId("add-note").click();
  await page.getByTestId("note-text-note1").fill("Try 32 filters in the first conv");
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("ControlOrMeta+f");      // centre on the group at 100 %
  await page.getByTestId("palette").getByRole("combobox").fill("@relu");
  await page.keyboard.press("Enter");
  await page.waitForTimeout(400);
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
  await page.locator(".canvas").screenshot({ path: `${OUT}/web_canvas_groups.png` });
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByTestId("palette").getByRole("combobox").fill("export");
  await page.screenshot({ path: `${OUT}/web_palette.png` });
});
