import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
const envTarget = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env
  ?.VITE_PROXY_TARGET;
const backendTarget = envTarget && envTarget.trim() ? envTarget.trim() : "http://127.0.0.1:8005";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      "/health": backendTarget,
      "/ui-config": backendTarget,
      "/stores": backendTarget,
      "/chat": backendTarget,
      "/sessions": backendTarget,
      "/history": backendTarget,
      "/embedding": backendTarget,
      "/evaluation": backendTarget
    }
  }
});
