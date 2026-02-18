import { useEffect, useMemo, useState } from "react";

import { ChatApiClient, OpsApiClient } from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { EmbeddingJobsPanel } from "./components/EmbeddingJobsPanel";
import { EvaluationPanel } from "./components/EvaluationPanel";
import { RepoStoreSelector } from "./components/RepoStoreSelector";
import { SessionSidebar } from "./components/SessionSidebar";
import { useChatSession } from "./hooks/useChatSession";
import { useEmbeddingJobs } from "./hooks/useEmbeddingJobs";
import { useEvaluation } from "./hooks/useEvaluation";
import type { UIConfigResponse } from "./types/api";


const DEFAULT_UI_CONFIG: UIConfigResponse = {
  title: "Vyom",
  subtitle: "AuditPilot AI - Code Auditor Assistant and Audit Finding Generator.",
  assistant_greeting: "Hi, I can help you analyze code, trace controls, and generate audit-ready findings.",
  input_placeholder: "Ask a question about the repository...",
  spinner_text: "Thinking...",
  show_sources: true,
  max_context_chunks: 10,
  history_messages: 20
};


function parseManualStores(raw: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const token of raw.split(",")) {
    const value = token.trim();
    if (!value || seen.has(value)) {
      continue;
    }
    seen.add(value);
    out.push(value);
  }
  return out;
}

function resolveChatApiBaseUrl(): string {
  return (import.meta.env.VITE_CHAT_API_BASE_URL ?? import.meta.env.VITE_API_BASE_URL ?? "").trim();
}

function resolveOpsApiBaseUrl(): string {
  return (import.meta.env.VITE_EMBEDDING_API_BASE_URL ?? import.meta.env.VITE_API_BASE_URL ?? "").trim();
}


export default function App() {
  const chatClient = useMemo(() => new ChatApiClient(resolveChatApiBaseUrl()), []);
  const opsClient = useMemo(() => new OpsApiClient(resolveOpsApiBaseUrl()), []);

  const [uiConfig, setUiConfig] = useState<UIConfigResponse>(DEFAULT_UI_CONFIG);
  const [stores, setStores] = useState<string[]>([]);
  const [selectedStores, setSelectedStores] = useState<string[]>([]);
  const [manualStoreInput, setManualStoreInput] = useState("");
  const [loadingStores, setLoadingStores] = useState(false);
  const [globalError, setGlobalError] = useState("");

  const chat = useChatSession(chatClient, uiConfig.assistant_greeting);
  const embedding = useEmbeddingJobs(opsClient);
  const evaluation = useEvaluation(opsClient);

  const effectiveStores = stores.length ? selectedStores : parseManualStores(manualStoreInput);

  const refreshStores = async () => {
    setLoadingStores(true);
    setGlobalError("");
    try {
      const nextStores = await chatClient.listStores();
      setStores(nextStores);
      if (nextStores.length) {
        setSelectedStores((prev) => {
          const filtered = prev.filter((item) => nextStores.includes(item));
          if (filtered.length) {
            return filtered;
          }
          return [nextStores[0]];
        });
      }
    } catch (exc) {
      setGlobalError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setLoadingStores(false);
    }
  };

  useEffect(() => {
    void (async () => {
      try {
        await chatClient.health();
      } catch (exc) {
        const detail = exc instanceof Error ? exc.message : String(exc);
        setGlobalError(
          `Chat API is unreachable (${detail}). Start chat service: python3 -m chat_module.api --config config.chat.yml`
        );
        return;
      }
      try {
        const cfg = await chatClient.getUIConfig();
        setUiConfig(cfg);
      } catch (exc) {
        setGlobalError(exc instanceof Error ? exc.message : String(exc));
      }
      await Promise.all([
        refreshStores(),
        chat.refreshSessions(),
        embedding.refreshJobs(),
        evaluation.refreshJobs(),
        evaluation.loadRules()
      ]);
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const onSend = async (message: string) => {
    if (!effectiveStores.length) {
      setGlobalError("Select at least one repo store before sending a message.");
      return;
    }
    setGlobalError("");
    await chat.sendMessage(message, effectiveStores);
  };

  const onNewSession = () => {
    chat.resetSession();
  };

  return (
    <div className="td-root">
      <div className="app-shell">
        <aside className="sidebar">
          <SessionSidebar
            currentSessionId={chat.sessionId}
            sessions={chat.sessions}
            onRefresh={chat.refreshSessions}
            onNewSession={onNewSession}
            onSelectSession={chat.loadSession}
            onDeleteSession={chat.removeSession}
          />
          <RepoStoreSelector
            stores={stores}
            selectedStores={selectedStores}
            manualStoreInput={manualStoreInput}
            onSelectStores={setSelectedStores}
            onManualInputChange={setManualStoreInput}
            onRefresh={refreshStores}
          />
          <EmbeddingJobsPanel
            jobs={embedding.jobs}
            loading={embedding.loading || loadingStores}
            error={embedding.error}
            onRefresh={embedding.refreshJobs}
            onStartJob={embedding.startJob}
          />
          <EvaluationPanel
            stores={stores}
            jobs={evaluation.jobs}
            rulesText={evaluation.rulesText}
            loadingJobs={evaluation.loadingJobs || loadingStores}
            loadingRules={evaluation.loadingRules}
            savingRules={evaluation.savingRules}
            startingJob={evaluation.startingJob}
            error={evaluation.error}
            onRefreshJobs={evaluation.refreshJobs}
            onLoadRules={evaluation.loadRules}
            onSaveRules={evaluation.saveRules}
            onRulesTextChange={evaluation.setRulesText}
            onImportRules={evaluation.importRulesFromFile}
            onExportRules={evaluation.exportRulesToFile}
            onStartJob={evaluation.startJob}
            htmlReportUrl={opsClient.getEvaluationReportHtmlUrl.bind(opsClient)}
            jsonReportUrl={opsClient.getEvaluationReportJsonUrl.bind(opsClient)}
          />
        </aside>
        <main className="content">
          {globalError || chat.error ? <div className="error-banner">{globalError || chat.error}</div> : null}
          <ChatPanel
            title={uiConfig.title}
            subtitle={uiConfig.subtitle}
            inputPlaceholder={uiConfig.input_placeholder}
            messages={chat.messages}
            sending={chat.sending || chat.loadingHistory}
            selectedStores={effectiveStores}
            onSend={onSend}
          />
        </main>
      </div>
    </div>
  );
}
