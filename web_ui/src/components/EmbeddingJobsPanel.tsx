import { FormEvent, useMemo, useState } from "react";

import type { EmbeddingJob } from "../types/api";


interface EmbeddingJobsPanelProps {
  jobs: EmbeddingJob[];
  loading: boolean;
  error: string;
  onRefresh: () => Promise<void>;
  onStartJob: (repoPath: string, repoName: string) => Promise<unknown>;
}


function defaultRepoName(repoPath: string): string {
  const normalized = repoPath.trim().replace(/[\\/]+$/, "");
  const segments = normalized.split(/[\\/]/).filter(Boolean);
  return segments[segments.length - 1] ?? "repo_store";
}

function fileNameOnly(pathValue: string): string {
  const normalized = pathValue.trim().replace(/[\\/]+$/, "");
  const segments = normalized.split(/[\\/]/).filter(Boolean);
  return segments[segments.length - 1] ?? pathValue;
}

function renderStatusCounts(job: EmbeddingJob): string {
  if (!job.file_status_counts) {
    return "";
  }
  return Object.entries(job.file_status_counts)
    .map(([key, value]) => `${key}=${value}`)
    .join(", ");
}

function JobItem({ job }: { job: EmbeddingJob }) {
  const counts = renderStatusCounts(job);
  return (
    <li className="job-item">
      <div className="job-id" title={job.job_id}>
        <strong>{job.job_id}</strong>
      </div>
      <div className="job-line">Status: {job.status}</div>
      {job.partial ? <div className="job-line">Partial: true</div> : null}
      <div className="job-line" title={job.repo_name}>Repo: {job.repo_name}</div>
      {counts ? (
        <div className="job-line" title={counts}>
          File status counts: {counts}
        </div>
      ) : null}
      {job.error ? (
        <div className="job-line error" title={job.error}>
          Error: {job.error}
        </div>
      ) : null}
      {job.log_path ? (
        <div className="job-line" title={job.log_path}>
          Log: {fileNameOnly(job.log_path)}
        </div>
      ) : null}
    </li>
  );
}


export function EmbeddingJobsPanel({
  jobs,
  loading,
  error,
  onRefresh,
  onStartJob
}: EmbeddingJobsPanelProps) {
  const [repoPath, setRepoPath] = useState("");
  const [repoName, setRepoName] = useState("");
  const [starting, setStarting] = useState(false);

  const computedRepoName = useMemo(() => (repoName.trim() ? repoName : defaultRepoName(repoPath)), [repoName, repoPath]);
  const currentJob = jobs.length ? jobs[0] : null;
  const olderJobs = jobs.slice(1);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!repoPath.trim() || !computedRepoName.trim()) {
      return;
    }
    setStarting(true);
    try {
      await onStartJob(repoPath.trim(), computedRepoName.trim());
      setRepoName("");
    } finally {
      setStarting(false);
    }
  };

  return (
    <section className="sidebar-card">
      <div className="sidebar-card-header">
        <h2>Embedding Jobs</h2>
        <button type="button" onClick={() => void onRefresh()}>
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </div>
      <form onSubmit={submit} className="embedding-form">
        <label>
          Repo path
          <input
            value={repoPath}
            onChange={(event) => setRepoPath(event.target.value)}
            placeholder="/path/to/repo"
            required
          />
        </label>
        <label>
          Repo store tag
          <input
            value={repoName}
            onChange={(event) => setRepoName(event.target.value)}
            placeholder={defaultRepoName(repoPath)}
          />
        </label>
        <button type="submit" disabled={starting || !repoPath.trim()}>
          {starting ? "Starting..." : "Start Embedding Job"}
        </button>
      </form>
      {error ? <p className="error">{error}</p> : null}
      <div className="current-job-wrap">
        <h3>Current Job</h3>
        {currentJob ? (
          <ul className="job-list">
            <JobItem job={currentJob} />
          </ul>
        ) : (
          <p className="job-empty">No embedding jobs yet.</p>
        )}
      </div>
      <details className="older-jobs">
        <summary>Older Jobs ({olderJobs.length})</summary>
        {olderJobs.length ? (
          <ul className="job-list">
            {olderJobs.map((job) => (
              <JobItem key={job.job_id} job={job} />
            ))}
          </ul>
        ) : (
          <p className="job-empty">No older jobs.</p>
        )}
      </details>
    </section>
  );
}
