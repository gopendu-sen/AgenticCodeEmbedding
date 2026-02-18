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
function parseTopLevelExtends(lines, filePath) {
    var _a, _b, _c, _d;
    var out = [];
    for (var index = 0; index < lines.length; index += 1) {
        var withoutComment = stripInlineComment(lines[index]);
        if (!withoutComment.trim()) {
            continue;
        }
        var indent = (_b = (_a = withoutComment.match(/^ */)) === null || _a === void 0 ? void 0 : _a[0].length) !== null && _b !== void 0 ? _b : 0;
        var depth = Math.floor(indent / 2);
        if (depth !== 0) {
            continue;
        }
        var trimmed = withoutComment.trim();
        if (!trimmed.startsWith("extends:")) {
            continue;
        }
        var rawValue = trimmed.slice("extends:".length).trim();
        if (rawValue) {
            var value = stripQuotes(rawValue);
            if (!value) {
                throw new Error("Invalid extends entry in ".concat(filePath));
            }
            out.push(value);
            return out;
        }
        for (var lookahead = index + 1; lookahead < lines.length; lookahead += 1) {
            var nestedRaw = stripInlineComment(lines[lookahead]);
            if (!nestedRaw.trim()) {
                continue;
            }
            var nestedIndent = (_d = (_c = nestedRaw.match(/^ */)) === null || _c === void 0 ? void 0 : _c[0].length) !== null && _d !== void 0 ? _d : 0;
            var nestedDepth = Math.floor(nestedIndent / 2);
            if (nestedDepth <= 0) {
                break;
            }
            var nestedTrimmed = nestedRaw.trim();
            if (nestedDepth === 1 && nestedTrimmed.startsWith("- ")) {
                var includeValue = stripQuotes(nestedTrimmed.slice(2).trim());
                if (!includeValue) {
                    throw new Error("Invalid extends list entry in ".concat(filePath));
                }
                out.push(includeValue);
                continue;
            }
            throw new Error("Invalid extends format in ".concat(filePath, ": expected list items under extends"));
        }
        return out;
    }
    return out;
}
function parseYamlScalars(filePath, ancestry) {
    var _a, _b;
    if (ancestry === void 0) { ancestry = new Set(); }
    var absolutePath = path.resolve(filePath);
    if (ancestry.has(absolutePath)) {
        throw new Error("Config extends cycle detected while reading ".concat(absolutePath));
    }
    var nextAncestry = new Set(ancestry);
    nextAncestry.add(absolutePath);
    var content = fs.readFileSync(absolutePath, "utf8");
    var lines = content.split(/\r?\n/);
    var out = {};
    var stack = [];
    for (var _i = 0, lines_1 = lines; _i < lines_1.length; _i++) {
        var rawLine = lines_1[_i];
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
    delete out["extends"];
    var merged = {};
    var extendsEntries = parseTopLevelExtends(lines, absolutePath);
    for (var _c = 0, extendsEntries_1 = extendsEntries; _c < extendsEntries_1.length; _c++) {
        var includePath = extendsEntries_1[_c];
        var resolvedInclude = path.resolve(path.dirname(absolutePath), includePath);
        Object.assign(merged, parseYamlScalars(resolvedInclude, nextAncestry));
    }
    Object.assign(merged, out);
    return merged;
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
