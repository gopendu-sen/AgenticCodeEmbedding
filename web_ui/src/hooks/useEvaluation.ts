import { useCallback, useState } from "react";

import { OpsApiClient } from "../api/client";
import type { EvaluationJob, EvaluationRulesPayload } from "../types/api";


function ensureRulesPayload(value: unknown): EvaluationRulesPayload {
  if (!value || typeof value !== "object") {
    throw new Error("Rules payload must be a JSON object");
  }
  const candidate = value as EvaluationRulesPayload;
  if (!Array.isArray(candidate.items)) {
    throw new Error("Rules payload must contain an items array");
  }
  return candidate;
}


export function useEvaluation(client: OpsApiClient) {
  const [jobs, setJobs] = useState<EvaluationJob[]>([]);
  const [rules, setRules] = useState<EvaluationRulesPayload | null>(null);
  const [rulesText, setRulesText] = useState<string>("");
  const [loadingJobs, setLoadingJobs] = useState(false);
  const [loadingRules, setLoadingRules] = useState(false);
  const [savingRules, setSavingRules] = useState(false);
  const [startingJob, setStartingJob] = useState(false);
  const [error, setError] = useState("");

  const refreshJobs = useCallback(
    async (limit = 20) => {
      setLoadingJobs(true);
      setError("");
      try {
        const next = await client.listEvaluationJobs(limit);
        setJobs(next);
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      } finally {
        setLoadingJobs(false);
      }
    },
    [client]
  );

  const loadRules = useCallback(async () => {
    setLoadingRules(true);
    setError("");
    try {
      const payload = await client.getEvaluationRules();
      setRules(payload);
      setRulesText(JSON.stringify(payload, null, 2));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setLoadingRules(false);
    }
  }, [client]);

  const saveRules = useCallback(async () => {
    setSavingRules(true);
    setError("");
    try {
      const parsed = JSON.parse(rulesText);
      const payload = ensureRulesPayload(parsed);
      const saved = await client.saveEvaluationRules(payload);
      setRules(saved);
      setRulesText(JSON.stringify(saved, null, 2));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      throw exc;
    } finally {
      setSavingRules(false);
    }
  }, [client, rulesText]);

  const importRulesFromFile = useCallback(async (file: File) => {
    setError("");
    const text = await file.text();
    const parsed = JSON.parse(text);
    const payload = ensureRulesPayload(parsed);
    setRules(payload);
    setRulesText(JSON.stringify(payload, null, 2));
  }, []);

  const exportRulesToFile = useCallback(() => {
    if (!rulesText.trim()) {
      throw new Error("Rules editor is empty");
    }
    const blob = new Blob([rulesText], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "evaluation_rules.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }, [rulesText]);

  const startJob = useCallback(
    async (repoName: string, repoPath: string) => {
      setStartingJob(true);
      setError("");
      try {
        const job = await client.startEvaluationJob(repoName, repoPath);
        await refreshJobs();
        return job;
      } catch (exc) {
        const message = exc instanceof Error ? exc.message : String(exc);
        setError(message);
        throw exc;
      } finally {
        setStartingJob(false);
      }
    },
    [client, refreshJobs]
  );

  return {
    jobs,
    rules,
    rulesText,
    loadingJobs,
    loadingRules,
    savingRules,
    startingJob,
    error,
    setError,
    setRulesText,
    refreshJobs,
    loadRules,
    saveRules,
    importRulesFromFile,
    exportRulesToFile,
    startJob,
  };
}
