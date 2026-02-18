// @ts-nocheck
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";


const processEnv = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env ?? {};
const configDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(configDir, "..");
const chatConfigPath = path.resolve(repoRoot, "config.chat.yml");
const embeddingConfigPath = path.resolve(repoRoot, "config.embedding.yml");
const legacyProxyTarget = processEnv.VITE_PROXY_TARGET;
const chatProxyTarget = processEnv.VITE_CHAT_PROXY_TARGET;
const opsProxyTarget = processEnv.VITE_EMBEDDING_PROXY_TARGET;
const guiHostOverride = (processEnv.VITE_GUI_HOST ?? "").trim();
const guiPortOverride = (processEnv.VITE_GUI_PORT ?? "").trim();


function stripInlineComment(line: string): string {
  let inSingle = false;
  let inDouble = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
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

function stripQuotes(value: string): string {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith("\"") && trimmed.endsWith("\"")) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function parseYamlScalars(filePath: string): Record<string, string> {
  const content = fs.readFileSync(filePath, "utf8");
  const out: Record<string, string> = {};
  const stack: string[] = [];

  for (const rawLine of content.split(/\r?\n/)) {
    const withoutComment = stripInlineComment(rawLine);
    if (!withoutComment.trim()) {
      continue;
    }

    const indent = withoutComment.match(/^ */)?.[0].length ?? 0;
    const depth = Math.floor(indent / 2);
    const trimmed = withoutComment.trim();

    if (trimmed.startsWith("- ")) {
      continue;
    }

    const colonIndex = trimmed.indexOf(":");
    if (colonIndex < 0) {
      continue;
    }

    const key = trimmed.slice(0, colonIndex).trim();
    const rawValue = trimmed.slice(colonIndex + 1).trim();

    while (stack.length > depth) {
      stack.pop();
    }

    if (!rawValue) {
      stack.push(key);
      continue;
    }

    const scalarPath = [...stack, key].join(".");
    out[scalarPath] = stripQuotes(rawValue);
  }

  return out;
}

function requireStringScalar(values: Record<string, string>, key: string, filePath: string): string {
  const value = (values[key] ?? "").trim();
  if (!value) {
    throw new Error(`Missing required '${key}' in ${filePath}`);
  }
  return value;
}

function requirePortScalar(values: Record<string, string>, key: string, filePath: string): number {
  const raw = requireStringScalar(values, key, filePath);
  const port = Number.parseInt(raw, 10);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`Invalid port '${raw}' for '${key}' in ${filePath}`);
  }
  return port;
}

function toProxyHost(host: string): string {
  const normalized = host.trim();
  if (normalized === "0.0.0.0" || normalized === "::") {
    return "127.0.0.1";
  }
  return normalized;
}

const chatConfig = parseYamlScalars(chatConfigPath);
const embeddingConfig = parseYamlScalars(embeddingConfigPath);

const chatConfigHost = requireStringScalar(chatConfig, "chat.api.host", chatConfigPath);
const chatConfigPort = requirePortScalar(chatConfig, "chat.api.port", chatConfigPath);
const opsConfigHost = requireStringScalar(embeddingConfig, "chat.api.host", embeddingConfigPath);
const opsConfigPort = requirePortScalar(embeddingConfig, "chat.api.port", embeddingConfigPath);
const guiConfigHost = requireStringScalar(chatConfig, "chat.ui.host", chatConfigPath);
const guiConfigPort = requirePortScalar(chatConfig, "chat.ui.port", chatConfigPath);

const chatTargetFromConfig = `http://${toProxyHost(chatConfigHost)}:${chatConfigPort}`;
const opsTargetFromConfig = `http://${toProxyHost(opsConfigHost)}:${opsConfigPort}`;

const guiHost = guiHostOverride || guiConfigHost;
const guiPort = guiPortOverride ? Number.parseInt(guiPortOverride, 10) : guiConfigPort;
if (!Number.isInteger(guiPort) || guiPort < 1 || guiPort > 65535) {
  throw new Error(`Invalid VITE_GUI_PORT value '${guiPortOverride}'`);
}

const chatTarget = chatProxyTarget && chatProxyTarget.trim()
  ? chatProxyTarget.trim()
  : legacyProxyTarget && legacyProxyTarget.trim()
    ? legacyProxyTarget.trim()
    : chatTargetFromConfig;

const opsTarget = opsProxyTarget && opsProxyTarget.trim() ? opsProxyTarget.trim() : opsTargetFromConfig;


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
