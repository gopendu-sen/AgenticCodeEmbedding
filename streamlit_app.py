import os
from typing import Any, Dict, List, Tuple

import streamlit as st

from agentic_rag.core.config import AgenticRagConfig
from agentic_rag.core.config_loader import load_agentic_rag_config
import retreiving_module.service as retriever_service_module
from retreiving_module import StoreRetriever


@st.cache_resource(show_spinner=False)
def _load_runtime(config_path: str, runtime_fingerprint: str) -> Tuple[AgenticRagConfig, StoreRetriever]:
    _ = runtime_fingerprint
    cfg = load_agentic_rag_config(config_path)
    retriever = StoreRetriever.from_config(cfg, config_path=config_path)
    return cfg, retriever


def _runtime_fingerprint(config_path: str) -> str:
    paths = [
        os.path.abspath(config_path),
        os.path.abspath(__file__),
        os.path.abspath(retriever_service_module.__file__),
    ]
    parts: List[str] = []
    for path in paths:
        try:
            mtime_ns = os.stat(path).st_mtime_ns
            parts.append(f"{path}:{mtime_ns}")
        except OSError:
            parts.append(f"{path}:missing")
    return "|".join(parts)


def _select_store_names(retriever: StoreRetriever) -> List[str]:
    refresh_clicked = st.sidebar.button("Refresh Repo Stores", use_container_width=True)
    if refresh_clicked or "repo_store_options" not in st.session_state:
        st.session_state["repo_store_options"] = retriever.discover_store_names()

    discovered = st.session_state.get("repo_store_options", [])
    if isinstance(discovered, list) and discovered:
        default_selected = st.session_state.get("repo_store_selected")
        if not isinstance(default_selected, list) or not default_selected:
            default_selected = discovered[:1]
        selected = st.sidebar.multiselect(
            "Repo Stores",
            options=discovered,
            default=default_selected,
            help="Select one or more repo tags to scope retrieval.",
        )
        st.session_state["repo_store_selected"] = selected
        return selected

    manual = st.sidebar.text_input(
        "Repo Stores (comma-separated)",
        value=st.session_state.get("repo_store_manual", ""),
        help="No discovered repo stores. Enter one or more repo tags manually, then click Refresh Repo Stores after embedding jobs complete.",
    )
    st.session_state["repo_store_manual"] = manual
    return retriever.parse_store_names(manual)


def _render_ingestion_panel(cfg: AgenticRagConfig, retriever: StoreRetriever) -> None:
    st.sidebar.markdown("### Embed Repository")
    repo_path = st.sidebar.text_input(
        "Repo path for embedding",
        value=st.session_state.get("embed_repo_path", cfg.paths.repo_path),
        help="Path to repository you want to index/refresh.",
    )
    st.session_state["embed_repo_path"] = repo_path
    repo_name = st.sidebar.text_input(
        "Repo store tag",
        value=st.session_state.get("embed_repo_name", retriever.default_repo_name(repo_path)),
        help="Tag used in metadata.repo_name for retrieval store selection.",
    )
    st.session_state["embed_repo_name"] = repo_name
    run_clicked = st.sidebar.button("Start Embedding Job", use_container_width=True)

    if run_clicked:
        try:
            job = retriever.start_embedding_job(repo_path=repo_path, repo_name=repo_name)
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Failed to start embedding job: {exc}")
            return
        st.sidebar.success(f"Embedding job started: {job.get('job_id')}")

    refresh_jobs_clicked = st.sidebar.button("Refresh Embedding Jobs", use_container_width=True)
    if refresh_jobs_clicked or "embedding_jobs" not in st.session_state:
        st.session_state["embedding_jobs"] = retriever.list_embedding_jobs(limit=10)

    jobs = st.session_state.get("embedding_jobs", [])
    if isinstance(jobs, list) and jobs:
        with st.sidebar.expander("Embedding Jobs", expanded=False):
            for job in jobs:
                job_id = job.get("job_id", "unknown")
                status = job.get("status", "unknown")
                repo = job.get("repo_name", "")
                st.write(f"- `{job_id}` status=`{status}` repo=`{repo}`")
                log_path = str(job.get("log_path", "")).strip()
                if log_path:
                    st.caption(f"log: `{log_path}`")
                if status == "completed" and isinstance(job.get("summary"), dict):
                    st.caption(
                        "changed="
                        + str(job["summary"].get("files_changed_indexed", ""))
                        + " embedded="
                        + str(job["summary"].get("nodes_embedded_upserted", ""))
                    )
                if status == "failed":
                    st.caption(f"error: {job.get('error', '')}")


def _ensure_session(cfg: AgenticRagConfig) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": cfg.chat.assistant_greeting, "sources": []}]


def _render_sources(cfg: AgenticRagConfig, sources: List[Dict[str, Any]]) -> None:
    if not (cfg.chat.show_sources and sources):
        return
    with st.expander("Sources", expanded=False):
        for src in sources:
            meta = src.get("metadata") or {}
            citation = src.get("citation_index", "?")
            st.write(
                f"- [{citation}] repo=`{meta.get('repo_name', 'unknown')}` "
                f"`{meta.get('file_path', 'unknown')}` "
                f"[{meta.get('start_line', '?')}-{meta.get('end_line', '?')}] "
                f"{meta.get('node_type', 'unknown')} "
                f"(collection={src.get('collection', 'unknown')}, distance={src.get('distance', 0):.4f})"
            )


def _render_chat(cfg: AgenticRagConfig) -> None:
    for message in st.session_state.messages:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(message.get("content", ""))
            _render_sources(cfg, message.get("sources") or [])


def main() -> None:
    st.set_page_config(page_title="Code RAG Chat", layout="wide")

    config_path = st.sidebar.text_input("Config path", value="config.yml")
    if st.sidebar.button("Reload Runtime", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    try:
        cfg, retriever = _load_runtime(config_path, _runtime_fingerprint(config_path))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load runtime: {exc}")
        st.stop()

    st.title(cfg.chat.title)
    st.caption(cfg.chat.subtitle)
    st.sidebar.caption(f"Retrieve log: `{retriever.retrieve_log_path}`")
    _render_ingestion_panel(cfg, retriever)

    store_names = _select_store_names(retriever)
    if not store_names:
        st.sidebar.warning("Select at least one repo store to enable retrieval.")

    _ensure_session(cfg)
    _render_chat(cfg)

    prompt = st.chat_input(cfg.chat.input_placeholder)
    if not prompt:
        return

    if not store_names:
        with st.chat_message("assistant"):
            st.warning("Select at least one repo store in the sidebar before sending prompts.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(cfg.chat.spinner_text):
            try:
                turn = retriever.chat_turn(
                    prompt=prompt,
                    history=st.session_state.messages,
                    store_names=store_names,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Chat retrieval failed: {exc}")
                return
        st.markdown(turn["response"])
        if not turn.get("sources"):
            st.info(
                "No indexed sources were retrieved for the selected repo store(s). "
                "Check `Embedding Jobs`, confirm the job completed, then click `Refresh Repo Stores`."
            )
        _render_sources(cfg, turn["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": turn["response"],
        "sources": turn["sources"],
    })


if __name__ == "__main__":
    main()
