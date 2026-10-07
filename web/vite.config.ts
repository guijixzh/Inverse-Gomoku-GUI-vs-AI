import { defineConfig } from "vite";

declare const process: { env: Record<string, string | undefined> };

export default defineConfig({
  base: process.env.VITE_BASE ?? "/",
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 2048,
  },
  worker: {
    format: "es",
  },
});
