import { FormEvent, useState } from "react";

import type { ChatMessage } from "../types/api";
import { SourcePanel } from "./SourcePanel";


interface ChatPanelProps {
  title: string;
  subtitle: string;
  inputPlaceholder: string;
  messages: ChatMessage[];
  sending: boolean;
  selectedStores: string[];
  onSend: (message: string) => Promise<void>;
}


export function ChatPanel({
  title,
  subtitle,
  inputPlaceholder,
  messages,
  sending,
  selectedStores,
  onSend
}: ChatPanelProps) {
  const [draft, setDraft] = useState("");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!draft.trim() || sending || !selectedStores.length) {
      return;
    }
    const payload = draft;
    setDraft("");
    await onSend(payload);
  };

  return (
    <section className="chat-panel">
      <header className="chat-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </header>
      <div className="message-list">
        {messages.map((message) => (
          <article key={message.id} className={`message message-${message.role}`}>
            <div className="message-role">{message.role}</div>
            <div className="message-content">{message.content}</div>
            {message.sources.length ? <SourcePanel sources={message.sources} /> : null}
          </article>
        ))}
      </div>
      <form className="chat-input" onSubmit={submit}>
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          rows={3}
          placeholder={inputPlaceholder}
          disabled={sending}
        />
        <button className="send-button" type="submit" disabled={sending || !draft.trim() || !selectedStores.length}>
          {sending ? "Generating..." : "Send Query"}
        </button>
      </form>
      {!selectedStores.length ? <p className="warning">Select at least one repo store to chat.</p> : null}
    </section>
  );
}
