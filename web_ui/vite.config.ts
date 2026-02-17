import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";


const processEnv = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env ?? {};
const legacyProxyTarget = processEnv.VITE_PROXY_TARGET;
const chatProxyTarget = processEnv.VITE_CHAT_PROXY_TARGET;
const opsProxyTarget = processEnv.VITE_EMBEDDING_PROXY_TARGET;

const chatTarget = chatProxyTarget && chatProxyTarget.trim()
  ? chatProxyTarget.trim()
  : legacyProxyTarget && legacyProxyTarget.trim()
    ? legacyProxyTarget.trim()
    : "http://127.0.0.1:8005";

const opsTarget = opsProxyTarget && opsProxyTarget.trim() ? opsProxyTarget.trim() : "http://127.0.0.1:8006";


export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      "/health": chatTarget,
      "/ui-config": chatTarget,
      "/stores": chatTarget,
      "/chat": chatTarget,
      "/sessions": chatTarget,
      "/history": chatTarget,
      "/embedding": opsTarget,
      "/evaluation": opsTarget,
      "/v1": opsTarget
    }
  }
});
