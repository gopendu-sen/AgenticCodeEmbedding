var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
var _a, _b, _c, _d;
// @ts-nocheck
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
var processEnv = (_b = (_a = globalThis.process) === null || _a === void 0 ? void 0 : _a.env) !== null && _b !== void 0 ? _b : {};
var configDir = path.dirname(fileURLToPath(import.meta.url));
var repoRoot = path.resolve(configDir, "..");
var chatConfigPath = path.resolve(repoRoot, "config.chat.yml");
var embeddingConfigPath = path.resolve(repoRoot, "config.embedding.yml");
var legacyProxyTarget = processEnv.VITE_PROXY_TARGET;
var chatProxyTarget = processEnv.VITE_CHAT_PROXY_TARGET;
var opsProxyTarget = processEnv.VITE_EMBEDDING_PROXY_TARGET;
var guiHostOverride = ((_c = processEnv.VITE_GUI_HOST) !== null && _c !== void 0 ? _c : "").trim();
var guiPortOverride = ((_d = processEnv.VITE_GUI_PORT) !== null && _d !== void 0 ? _d : "").trim();
function stripInlineComment(line) {
    var inSingle = false;
    var inDouble = false;
    for (var index = 0; index < line.length; index += 1) {
        var char = line[index];
        if (char === "'" && !inDouble) {
            inSingle = !inSingle;
            continue;
        }
        if (char === "\"" && !inSingle) {
            inDouble = !inDouble;
            continue;
        }
        if (char === "#" && !inSingle && !inDouble) {
            return line.slice(0, index);
        }
    }
    return line;
}
function stripQuotes(value) {
    var trimmed = value.trim();
    if ((trimmed.startsWith("\"") && trimmed.endsWith("\"")) ||
        (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
        return trimmed.slice(1, -1);
    }
    return trimmed;
}
function parseYamlScalars(filePath) {
    var _a, _b;
    var content = fs.readFileSync(filePath, "utf8");
    var out = {};
    var stack = [];
    for (var _i = 0, _c = content.split(/\r?\n/); _i < _c.length; _i++) {
        var rawLine = _c[_i];
        var withoutComment = stripInlineComment(rawLine);
        if (!withoutComment.trim()) {
            continue;
        }
        var indent = (_b = (_a = withoutComment.match(/^ */)) === null || _a === void 0 ? void 0 : _a[0].length) !== null && _b !== void 0 ? _b : 0;
        var depth = Math.floor(indent / 2);
        var trimmed = withoutComment.trim();
        if (trimmed.startsWith("- ")) {
            continue;
        }
        var colonIndex = trimmed.indexOf(":");
        if (colonIndex < 0) {
            continue;
        }
        var key = trimmed.slice(0, colonIndex).trim();
        var rawValue = trimmed.slice(colonIndex + 1).trim();
        while (stack.length > depth) {
            stack.pop();
        }
        if (!rawValue) {
            stack.push(key);
            continue;
        }
        var scalarPath = __spreadArray(__spreadArray([], stack, true), [key], false).join(".");
        out[scalarPath] = stripQuotes(rawValue);
    }
    return out;
}
function requireStringScalar(values, key, filePath) {
    var _a;
    var value = ((_a = values[key]) !== null && _a !== void 0 ? _a : "").trim();
    if (!value) {
        throw new Error("Missing required '".concat(key, "' in ").concat(filePath));
    }
    return value;
}
function requirePortScalar(values, key, filePath) {
    var raw = requireStringScalar(values, key, filePath);
    var port = Number.parseInt(raw, 10);
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
        throw new Error("Invalid port '".concat(raw, "' for '").concat(key, "' in ").concat(filePath));
    }
    return port;
}
function toProxyHost(host) {
    var normalized = host.trim();
    if (normalized === "0.0.0.0" || normalized === "::") {
        return "127.0.0.1";
    }
    return normalized;
}
var chatConfig = parseYamlScalars(chatConfigPath);
var embeddingConfig = parseYamlScalars(embeddingConfigPath);
var chatConfigHost = requireStringScalar(chatConfig, "chat.api.host", chatConfigPath);
var chatConfigPort = requirePortScalar(chatConfig, "chat.api.port", chatConfigPath);
var opsConfigHost = requireStringScalar(embeddingConfig, "chat.api.host", embeddingConfigPath);
var opsConfigPort = requirePortScalar(embeddingConfig, "chat.api.port", embeddingConfigPath);
var guiConfigHost = requireStringScalar(chatConfig, "chat.ui.host", chatConfigPath);
var guiConfigPort = requirePortScalar(chatConfig, "chat.ui.port", chatConfigPath);
var chatTargetFromConfig = "http://".concat(toProxyHost(chatConfigHost), ":").concat(chatConfigPort);
var opsTargetFromConfig = "http://".concat(toProxyHost(opsConfigHost), ":").concat(opsConfigPort);
var guiHost = guiHostOverride || guiConfigHost;
var guiPort = guiPortOverride ? Number.parseInt(guiPortOverride, 10) : guiConfigPort;
if (!Number.isInteger(guiPort) || guiPort < 1 || guiPort > 65535) {
    throw new Error("Invalid VITE_GUI_PORT value '".concat(guiPortOverride, "'"));
}
var chatTarget = chatProxyTarget && chatProxyTarget.trim()
    ? chatProxyTarget.trim()
    : legacyProxyTarget && legacyProxyTarget.trim()
        ? legacyProxyTarget.trim()
        : chatTargetFromConfig;
var opsTarget = opsProxyTarget && opsProxyTarget.trim() ? opsProxyTarget.trim() : opsTargetFromConfig;
export default defineConfig({
    plugins: [react()],
    server: {
        host: guiHost,
        port: guiPort,
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
