import type { Page } from "@playwright/test";

/** Open the File menu and pick an item by its test id ("open-project", "new-project") or label. */
export async function fileMenu(page: Page, item: string): Promise<void> {
  await page.getByTestId("file-menu").click();
  const menu = page.getByRole("menu", { name: "File" });
  const byId = menu.getByTestId(item);
  if (await byId.count()) await byId.click();
  else await menu.getByRole("menuitem", { name: item }).click();
}
