var _a, _b;
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
var processEnv = (_b = (_a = globalThis.process) === null || _a === void 0 ? void 0 : _a.env) !== null && _b !== void 0 ? _b : {};
var legacyProxyTarget = processEnv.VITE_PROXY_TARGET;
var chatProxyTarget = processEnv.VITE_CHAT_PROXY_TARGET;
var opsProxyTarget = processEnv.VITE_EMBEDDING_PROXY_TARGET;
var chatTarget = chatProxyTarget && chatProxyTarget.trim()
    ? chatProxyTarget.trim()
    : legacyProxyTarget && legacyProxyTarget.trim()
        ? legacyProxyTarget.trim()
        : "http://127.0.0.1:8005";
var opsTarget = opsProxyTarget && opsProxyTarget.trim() ? opsProxyTarget.trim() : "http://127.0.0.1:8006";
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
