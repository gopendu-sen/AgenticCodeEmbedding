import { useCallback, useState } from "react";

import { OpsApiClient } from "../api/client";
import type { EmbeddingJob } from "../types/api";


export function useEmbeddingJobs(client: OpsApiClient) {
  const [jobs, setJobs] = useState<EmbeddingJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const refreshJobs = useCallback(
    async (limitOrEvent?: unknown) => {
      const limit =
        typeof limitOrEvent === "number" && Number.isFinite(limitOrEvent) && limitOrEvent > 0
          ? Math.floor(limitOrEvent)
          : 20;
      setLoading(true);
      setError("");
      try {
        const next = await client.listEmbeddingJobs(limit);
        setJobs(next);
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      } finally {
        setLoading(false);
      }
    },
    [client]
  );

  const startJob = useCallback(
    async (repoPath: string, repoName: string) => {
      setError("");
      try {
        const job = await client.startEmbeddingJob(repoPath, repoName);
        await refreshJobs();
        return job;
      } catch (exc) {
        const message = exc instanceof Error ? exc.message : String(exc);
        setError(message);
        throw exc;
      }
    },
    [client, refreshJobs]
  );

  return {
    jobs,
    loading,
    error,
    refreshJobs,
    startJob
  };
}
