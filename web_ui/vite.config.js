var _a, _b;
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
var envTarget = (_b = (_a = globalThis.process) === null || _a === void 0 ? void 0 : _a.env) === null || _b === void 0 ? void 0 : _b.VITE_PROXY_TARGET;
var backendTarget = envTarget && envTarget.trim() ? envTarget.trim() : "http://127.0.0.1:8005";
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
