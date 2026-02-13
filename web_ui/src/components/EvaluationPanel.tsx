import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

import type { EvaluationJob } from "../types/api";


interface EvaluationPanelProps {
  stores: string[];
  jobs: EvaluationJob[];
  rulesText: string;
  loadingJobs: boolean;
  loadingRules: boolean;
  savingRules: boolean;
  startingJob: boolean;
  error: string;
  onRefreshJobs: () => Promise<void>;
  onLoadRules: () => Promise<void>;
  onSaveRules: () => Promise<void>;
  onRulesTextChange: (value: string) => void;
  onImportRules: (file: File) => Promise<void>;
  onExportRules: () => void;
  onStartJob: (repoName: string, repoPath: string) => Promise<unknown>;
  htmlReportUrl: (jobId: string) => string;
  jsonReportUrl: (jobId: string) => string;
}


function fileNameOnly(pathValue: string): string {
  const normalized = pathValue.trim().replace(/[\\/]+$/, "");
  const segments = normalized.split(/[\\/]/).filter(Boolean);
  return segments[segments.length - 1] ?? pathValue;
}

function renderCounts(job: EvaluationJob): string {
  const counts = job.status_counts;
  if (!counts) {
    return "";
  }
  return `detected=${counts.detected ?? 0}, not_detected=${counts.not_detected ?? 0}, needs_review=${counts.needs_review ?? 0}`;
}

function JobItem({
  job,
  htmlReportUrl,
  jsonReportUrl
}: {
  job: EvaluationJob;
  htmlReportUrl: (jobId: string) => string;
  jsonReportUrl: (jobId: string) => string;
}) {
  const countsText = renderCounts(job);
  const hasReports = Boolean(job.report_html_path || job.report_json_path);
  return (
    <li className="job-item">
      <div className="job-id" title={job.job_id}>
        <strong>{job.job_id}</strong>
      </div>
      <div className="job-line">Status: {job.status}</div>
      {job.partial ? <div className="job-line">Partial: true</div> : null}
      <div className="job-line" title={job.repo_name}>Repo: {job.repo_name}</div>
      {typeof job.rule_count === "number" ? <div className="job-line">Rules: {job.rule_count}</div> : null}
      {countsText ? (
        <div className="job-line" title={countsText}>
          Status counts: {countsText}
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
      {hasReports && (job.status === "completed" || job.status === "completed_partial") ? (
        <div className="job-actions">
          <a href={htmlReportUrl(job.job_id)} target="_blank" rel="noreferrer">
            Open HTML
          </a>
          <a href={jsonReportUrl(job.job_id)} target="_blank" rel="noreferrer">
            Download JSON
          </a>
        </div>
      ) : null}
    </li>
  );
}


export function EvaluationPanel({
  stores,
  jobs,
  rulesText,
  loadingJobs,
  loadingRules,
  savingRules,
  startingJob,
  error,
  onRefreshJobs,
  onLoadRules,
  onSaveRules,
  onRulesTextChange,
  onImportRules,
  onExportRules,
  onStartJob,
  htmlReportUrl,
  jsonReportUrl
}: EvaluationPanelProps) {
  const [repoPath, setRepoPath] = useState("");
  const [selectedStore, setSelectedStore] = useState("");
  const [manualStore, setManualStore] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (stores.length > 0) {
      setSelectedStore((prev) => (prev && stores.includes(prev) ? prev : stores[0]));
    }
  }, [stores]);

  const currentJob = jobs.length ? jobs[0] : null;
  const olderJobs = jobs.slice(1);
  const effectiveStore = useMemo(() => {
    if (stores.length > 0) {
      return selectedStore.trim();
    }
    return manualStore.trim();
  }, [manualStore, selectedStore, stores.length]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!repoPath.trim() || !effectiveStore) {
      return;
    }
    await onStartJob(effectiveStore, repoPath.trim());
  };

  const onImportChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    await onImportRules(file);
    event.target.value = "";
  };

  return (
    <section className="sidebar-card">
      <div className="sidebar-card-header">
        <h2>Evaluation Reports</h2>
        <button type="button" onClick={() => void onRefreshJobs()}>
          {loadingJobs ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      <form onSubmit={submit} className="embedding-form">
        {stores.length > 0 ? (
          <label>
            Repo store
            <select value={selectedStore} onChange={(event) => setSelectedStore(event.target.value)}>
              {stores.map((store) => (
                <option key={store} value={store}>
                  {store}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <label>
            Repo store
            <input
              value={manualStore}
              onChange={(event) => setManualStore(event.target.value)}
              placeholder="repo_store_name"
              required
            />
          </label>
        )}
        <label>
          Repo path
          <input
            value={repoPath}
            onChange={(event) => setRepoPath(event.target.value)}
            placeholder="/path/to/repo"
            required
          />
        </label>
        <button type="submit" disabled={startingJob || !repoPath.trim() || !effectiveStore}>
          {startingJob ? "Starting..." : "Start Evaluation Job"}
        </button>
      </form>

      {error ? <p className="error">{error}</p> : null}

      <div className="current-job-wrap">
        <h3>Current Evaluation Job</h3>
        {currentJob ? (
          <ul className="job-list">
            <JobItem job={currentJob} htmlReportUrl={htmlReportUrl} jsonReportUrl={jsonReportUrl} />
          </ul>
        ) : (
          <p className="job-empty">No evaluation jobs yet.</p>
        )}
      </div>
      <details className="older-jobs">
        <summary>Older Evaluation Jobs ({olderJobs.length})</summary>
        {olderJobs.length ? (
          <ul className="job-list">
            {olderJobs.map((job) => (
              <JobItem key={job.job_id} job={job} htmlReportUrl={htmlReportUrl} jsonReportUrl={jsonReportUrl} />
            ))}
          </ul>
        ) : (
          <p className="job-empty">No older evaluation jobs.</p>
        )}
      </details>

      <details className="older-jobs">
        <summary>Evaluation Rules JSON</summary>
        <div className="rules-actions">
          <button type="button" onClick={() => void onLoadRules()}>
            {loadingRules ? "Loading..." : "Load"}
          </button>
          <button type="button" onClick={() => void onSaveRules()} disabled={savingRules || !rulesText.trim()}>
            {savingRules ? "Saving..." : "Save"}
          </button>
          <button type="button" onClick={() => fileInputRef.current?.click()}>
            Import
          </button>
          <button type="button" onClick={onExportRules} disabled={!rulesText.trim()}>
            Export
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/json"
            className="hidden-file-input"
            onChange={(event) => void onImportChange(event)}
          />
        </div>
        <textarea
          className="rules-editor"
          value={rulesText}
          onChange={(event) => onRulesTextChange(event.target.value)}
          rows={16}
          placeholder="Evaluation rules JSON"
        />
      </details>
    </section>
  );
}
