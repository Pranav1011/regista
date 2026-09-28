import { defineConfig } from "vite";

// Relative base so the static build works from any path (e.g. GitHub Pages /regista/).
export default defineConfig({
  base: "./",
  build: { target: "es2022", sourcemap: false },
});
