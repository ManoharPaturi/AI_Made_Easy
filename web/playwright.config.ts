import { defineConfig, devices } from "@playwright/test";

// Starts `aime web` against a throwaway $AIME_HOME and runs the smoke tests in Chromium.
const port = Number(process.env.AIME_E2E_PORT ?? 8799);
const home = process.env.AIME_E2E_HOME ?? `${process.env.TMPDIR ?? "/tmp"}/aime-e2e-${Date.now()}`;

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: {
    command: `${process.env.AIME_BIN ?? "aime"} web --port ${port}`,
    url: `http://127.0.0.1:${port}/api/info`,
    env: { AIME_HOME: home, QT_QPA_PLATFORM: "offscreen" },
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
