// Documentation screenshots of the web UI.
// Usage: aime web --port 8798 &  node scripts/screenshots.mjs [baseURL] [outDir]
import { chromium } from "@playwright/test";

const base = process.argv[2] ?? "http://127.0.0.1:8798";
const out = process.argv[3] ?? "../docs/images";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 960 }, deviceScaleFactor: 1 });

await page.goto(base);
await page.getByPlaceholder(/Search \d{3} blocks/).waitFor();
await page.locator(".block-node").first().waitFor();
await page.getByTestId("status").filter({ hasText: "No problems" }).waitFor();
await page.getByRole("button", { name: "Open" }).click();
await page.getByTestId("sample-iris_mlp.json").click();
await page.getByTestId("status").filter({ hasText: "No problems" }).waitFor();
await page.waitForTimeout(500);
await page.locator('[data-testid^="node-dense"]').first().click();
await page.waitForTimeout(600);
await page.screenshot({ path: `${out}/web_designer.png` });

await page.evaluate(() => localStorage.setItem("aime.theme", "light"));
await page.reload();
await page.locator(".block-node").first().waitFor();
await page.waitForTimeout(500);
await page.getByRole("button", { name: "Code" }).click();
await page.getByTestId("code-view").waitFor();
await page.waitForTimeout(600);
await page.screenshot({ path: `${out}/web_designer_light.png` });
await browser.close();
console.log(`saved screenshots to ${out}`);
