import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Two entry points, one build. `base: "./"` is load-bearing: pywebview opens
// the bundle from a file:// path, so every asset URL must be relative or the
// window renders blank with no error anywhere.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        data: resolve(__dirname, "data.html"),
        agents: resolve(__dirname, "agents.html"),
      },
    },
  },
});
