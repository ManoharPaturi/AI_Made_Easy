import { expect, test } from "@playwright/test";
import { fileMenu } from "./helpers";

// A 500-block design opens, validates and takes an edit without stalling the page.
function chain(n: number) {
  const nodes: object[] = [{ id: "in", type: "core.input", params: { shape: "64", dtype: "float32" },
                             position: [0, 0] }];
  const edges: object[] = [];
  let prev = "in";
  for (let i = 0; i < n; i++) {
    const dense = i % 2 === 0;
    nodes.push({ id: `n${i}`, type: dense ? "core.dense" : "core.relu",
                 params: dense ? { units: 64 } : {},
                 position: [((i + 1) % 25) * 250, Math.floor((i + 1) / 25) * 150] });
    edges.push({ from: `${prev}/out`, to: `n${i}/in` });
    prev = `n${i}`;
  }
  nodes.push({ id: "out", type: "core.output", params: {}, position: [((n + 1) % 25) * 250, Math.floor((n + 1) / 25) * 150] });
  edges.push({ from: `${prev}/out`, to: "out/in" });
  return { schema_version: 3, name: "e2e_large", nodes, edges, meta: {} };
}

test("a 500-block design opens and validates quickly", async ({ page, request }) => {
  const put = await request.put("/api/projects/e2e_large", { data: chain(500) });
  expect(put.ok(), await put.text()).toBeTruthy();
  try {
    await page.goto("/");
    await expect(page.locator(".block-node").first()).toBeVisible();
    await fileMenu(page, "open-project");
    const start = Date.now();
    await page.locator(".issue", { hasText: "e2e_large" }).click();
    await expect(page.locator(".block-node")).toHaveCount(502, { timeout: 30_000 });
    await expect(page.getByTestId("status")).toHaveText("No problems", { timeout: 30_000 });
    const opened = Date.now() - start;
    console.log(`opened 500 blocks in ${opened} ms`);
    expect(opened).toBeLessThan(15_000);
    // an edit revalidates the whole design
    // the first dense block on screen (a 500-block design is mostly off-screen)
    const box = await page.getByTestId("canvas").boundingBox();
    const dense = page.locator(".block-node", { hasText: "units 64" });
    let target = dense.first();
    for (let i = 0; i < await dense.count(); i++) {
      const b = await dense.nth(i).boundingBox();
      if (b && box && b.x > box.x && b.y > box.y && b.x + b.width < box.x + box.width
          && b.y + b.height < box.y + box.height) { target = dense.nth(i); break; }
    }
    const edit = Date.now();
    await target.click();
    await page.locator("#param-units").fill("0");
    await page.locator("#param-units").press("Enter");
    await expect(page.getByTestId("status")).not.toHaveText("No problems", { timeout: 30_000 });
    console.log(`revalidated after an edit in ${Date.now() - edit} ms`);
    expect(Date.now() - edit).toBeLessThan(10_000);
  } finally {
    await request.delete("/api/projects/e2e_large");
  }
});
