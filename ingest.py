import argparse
import logging

from agentic_rag.agentic_ai.orchestrator import AgenticRagOrchestrator
from agentic_rag.core.config_loader import load_agentic_rag_config


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Vyom ingestion")
    parser.add_argument("--config", default="config.yml", help="Path to config YAML file")
    parser.add_argument("--repo-name", required=True, help="Repo store tag used for embedding metadata")
    args = parser.parse_args()

    cfg = load_agentic_rag_config(args.config)
    level_name = (cfg.logging.level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info(
        "Config loaded: config_path=%s repo=%s repo_name=%s llm_model=%s embedding_model=%s",
        args.config,
        cfg.paths.repo_path,
        args.repo_name,
        cfg.llm.model,
        cfg.embedding.model,
    )

    orch = AgenticRagOrchestrator(cfg, repo_name=args.repo_name)
    logger.info("Starting ingestion run")
    summary = orch.run()
    logger.info("Ingestion run completed")

    print("\n=== Vyom Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
