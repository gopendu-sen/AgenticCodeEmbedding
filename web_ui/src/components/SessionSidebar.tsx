import type { SessionSummary } from "../types/api";


interface SessionSidebarProps {
  currentSessionId: string;
  sessions: SessionSummary[];
  onRefresh: () => Promise<void>;
  onNewSession: () => void;
  onSelectSession: (sessionId: string) => Promise<void>;
  onDeleteSession: (sessionId: string) => Promise<void>;
}

function truncateOneLine(value: string, maxChars: number): string {
  const single = value.replace(/\s+/g, " ").trim();
  if (single.length <= maxChars) {
    return single;
  }
  return `${single.slice(0, Math.max(0, maxChars - 1))}…`;
}

function shortSessionId(sessionId: string): string {
  if (sessionId.length <= 16) {
    return sessionId;
  }
  return `${sessionId.slice(0, 6)}…${sessionId.slice(-6)}`;
}

function formatUpdatedAt(unixSeconds: number): string {
  if (!Number.isFinite(unixSeconds) || unixSeconds <= 0) {
    return "";
  }
  const date = new Date(unixSeconds * 1000);
  return date.toLocaleString();
}


export function SessionSidebar({
  currentSessionId,
  sessions,
  onRefresh,
  onNewSession,
  onSelectSession,
  onDeleteSession
}: SessionSidebarProps) {
  return (
    <section className="sidebar-card">
      <div className="sidebar-card-header">
        <h2>Sessions</h2>
        <div className="session-actions">
          <button type="button" onClick={() => void onRefresh()}>
            Refresh
          </button>
          <button type="button" onClick={onNewSession}>
            New
          </button>
        </div>
      </div>
      <ul className="session-list">
        {sessions.map((session) => {
          const isActive = session.session_id === currentSessionId;
          return (
            <li key={session.session_id} className={isActive ? "active" : ""}>
              <button type="button" onClick={() => void onSelectSession(session.session_id)}>
                <div className="session-id" title={session.session_id}>
                  {shortSessionId(session.session_id)}
                </div>
                <div className="session-meta">
                  {session.message_count} msgs
                  {session.updated_at ? ` • ${formatUpdatedAt(session.updated_at)}` : ""}
                </div>
                <div className="session-last" title={session.last_message || "(no messages yet)"}>
                  {truncateOneLine(session.last_message || "(no messages yet)", 88)}
                </div>
              </button>
              <button
                type="button"
                className="delete-session"
                onClick={() => void onDeleteSession(session.session_id)}
              >
                Delete
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
