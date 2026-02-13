import type { SourceChunk } from "../types/api";


interface SourcePanelProps {
  sources: SourceChunk[];
}


export function SourcePanel({ sources }: SourcePanelProps) {
  if (!sources.length) {
    return null;
  }
  return (
    <details className="source-panel">
      <summary>Sources ({sources.length})</summary>
      <ul>
        {sources.map((src, index) => {
          const meta = src.metadata ?? {};
          const citation = src.citation_index ?? index + 1;
          const start = meta.start_line ?? "?";
          const end = meta.end_line ?? "?";
          return (
            <li key={`${citation}-${index}`}>
              <div className="source-head">
                <strong>[{citation}]</strong>{" "}
                <code>{meta.repo_name ?? "unknown_repo"}</code>{" "}
                <code>{meta.file_path ?? "unknown_file"}</code>{" "}
                <span>
                  [{start}-{end}]
                </span>{" "}
                <span>{meta.node_type ?? "unknown_type"}</span>{" "}
                <span>collection={src.collection ?? "unknown"}</span>{" "}
                <span>
                  distance=
                  {typeof src.distance === "number" ? src.distance.toFixed(4) : "?"}
                </span>
              </div>
              {src.document ? <pre>{src.document}</pre> : null}
            </li>
          );
        })}
      </ul>
    </details>
  );
}
