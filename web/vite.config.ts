import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies the API to `aime web --port 8765`; `npm run build` writes the
// bundle into the Python package so pip installs serve it without Node.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../ai_made_easy/server/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8765", ws: true },
    },
  },
});
