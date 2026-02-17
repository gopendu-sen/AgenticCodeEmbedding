import { useCallback, useEffect, useMemo, useState } from "react";

import { ChatApiClient } from "../api/client";
import type {
  ChatDoneEvent,
  ChatEvent,
  ChatMessage,
  SessionSummary,
  SourceChunk
} from "../types/api";


function newSessionId(): string {
  return `sess_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
}


function greetingMessage(greeting: string): ChatMessage {
  return {
    id: "assistant_greeting",
    role: "assistant",
    content: greeting,
    sources: [],
    citedIndices: [],
    intent: "",
    createdAt: Date.now() / 1000
  };
}


export function useChatSession(client: ChatApiClient, assistantGreeting: string) {
  const [sessionId, setSessionId] = useState<string>(() => newSessionId());
  const [messages, setMessages] = useState<ChatMessage[]>(() => [greetingMessage(assistantGreeting)]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  const hasOnlyGreeting = useMemo(
    () => messages.length === 1 && messages[0]?.id === "assistant_greeting",
    [messages]
  );

  useEffect(() => {
    setMessages((prev) => {
      if (prev.length !== 1 || prev[0]?.id !== "assistant_greeting") {
        return prev;
      }
      return [greetingMessage(assistantGreeting)];
    });
  }, [assistantGreeting]);

  const resetSession = useCallback(
    (nextSessionId?: string) => {
      setSessionId(nextSessionId ?? newSessionId());
      setMessages([greetingMessage(assistantGreeting)]);
      setError("");
    },
    [assistantGreeting]
  );

  const refreshSessions = useCallback(async () => {
    try {
      const payload = await client.listSessions(100);
      setSessions(payload);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, [client]);

  const loadSession = useCallback(
    async (targetSessionId: string) => {
      setLoadingHistory(true);
      setError("");
      try {
        const history = await client.getHistory(targetSessionId);
        setSessionId(targetSessionId);
        if (!history.messages.length) {
          setMessages([greetingMessage(assistantGreeting)]);
          return;
        }
        setMessages(
          history.messages.map((item) => ({
            id: String(item.id),
            role: item.role,
            content: item.content,
            sources: item.sources ?? [],
            citedIndices: item.cited_indices ?? [],
            intent: item.intent ?? "",
            createdAt: item.created_at
          }))
        );
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      } finally {
        setLoadingHistory(false);
      }
    },
    [assistantGreeting, client]
  );

  const removeSession = useCallback(
    async (targetSessionId: string) => {
      try {
        await client.deleteSession(targetSessionId);
        if (targetSessionId === sessionId) {
          resetSession();
        }
        await refreshSessions();
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      }
    },
    [client, refreshSessions, resetSession, sessionId]
  );

  const sendMessage = useCallback(
    async (message: string, storeNames: string[]) => {
      const trimmed = message.trim();
      if (!trimmed || sending) {
        return;
      }
      setSending(true);
      setError("");

      const userMsg: ChatMessage = {
        id: `user_${Date.now()}`,
        role: "user",
        content: trimmed,
        sources: [],
        citedIndices: [],
        intent: "",
        createdAt: Date.now() / 1000
      };
      const assistantId = `assistant_${Date.now()}`;
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        sources: [],
        citedIndices: [],
        intent: "",
        createdAt: Date.now() / 1000
      };

      setMessages((prev) => {
        const base = hasOnlyGreeting ? [] : prev;
        return [...base, userMsg, assistantMsg];
      });

      let latestSources: SourceChunk[] = [];
      let latestIntent = "";
      let latestCitations: number[] = [];

      try {
        await client.streamChat(
          {
            session_id: sessionId,
            message: trimmed,
            store_names: storeNames
          },
          (event: ChatEvent) => {
            if (event.event === "meta") {
              latestIntent = event.intent;
              latestSources = event.sources ?? [];
              return;
            }
            if (event.event === "token") {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId ? { ...msg, content: `${msg.content}${event.token}` } : msg
                )
              );
              return;
            }
            if (event.event === "done") {
              const done = event as ChatDoneEvent;
              latestIntent = done.intent ?? latestIntent;
              latestSources = done.sources ?? latestSources;
              latestCitations = done.cited_indices ?? [];
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? {
                        ...msg,
                        content: done.response ?? msg.content,
                        sources: latestSources,
                        citedIndices: latestCitations,
                        intent: latestIntent
                      }
                    : msg
                )
              );
              return;
            }
            if (event.event === "error") {
              const errText = event.message || "Chat stream failed";
              setError(errText);
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, content: errText, sources: latestSources, citedIndices: [], intent: latestIntent }
                    : msg
                )
              );
            }
          }
        );
      } catch (exc) {
        const errText = exc instanceof Error ? exc.message : String(exc);
        setError(errText);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: errText, sources: latestSources, citedIndices: [], intent: latestIntent }
              : msg
          )
        );
      } finally {
        setSending(false);
        await refreshSessions();
      }
    },
    [client, hasOnlyGreeting, refreshSessions, sending, sessionId]
  );

  return {
    sessionId,
    messages,
    sessions,
    loadingHistory,
    sending,
    error,
    setError,
    hasOnlyGreeting,
    refreshSessions,
    loadSession,
    sendMessage,
    resetSession,
    removeSession
  };
}
