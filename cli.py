# cli.py

"""
Command-line interface for the extraction worker.

This module provides a thin operational wrapper around run_worker() with
no business logic. Supports both CLI arguments and environment variables
for AWS/headless deployment.
"""

import sys

import click
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

from db.db import get_db_engine
from logging_configuration import configure_logging, get_logger
from text_extraction.basic_extraction import TextFileTextExtractor
from text_extraction.pdf_extraction import PDFTextExtractor
from worker import run_worker


@click.command()
@click.option(
    "--once",
    is_flag=True,
    envvar="ONCE",
    help="Process one batch then exit",
)
@click.option(
    "--limit",
    type=int,
    default=10,
    envvar="LIMIT",
    show_default=True,
    help="Maximum files to process per batch",
)
@click.option(
    "--poll-seconds",
    type=float,
    default=5.0,
    envvar="POLL_SECONDS",
    show_default=True,
    help="Seconds between batch polls",
)
@click.option(
    "--extensions",
    type=str,
    envvar="EXTENSIONS",
    help="Comma-separated file extensions to process",
)
@click.option(
    "--max-chars",
    type=int,
    envvar="MAX_CHARS",
    help="Maximum characters to extract. Files exceeding this limit will be recorded as failures and skipped.",
)
@click.option(
    "--embed/--no-embed",
    "enable_embedding",
    default=True,
    envvar="ENABLE_EMBEDDING",
    show_default=True,
    help="Enable/disable embedding generation",
)
@click.option(
    "--embedder",
    type=click.Choice(["minilm"], case_sensitive=False),
    default="minilm",
    envvar="EMBEDDER",
    show_default=True,
    help="Embedder model to use",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    envvar="LOG_LEVEL",
    show_default=True,
    help="Logging level",
)
@click.option(
    "--log-file",
    type=click.Path(),
    envvar="LOG_FILE",
    help="Path to log file",
)
@click.option(
    "--json-logs",
    is_flag=True,
    envvar="JSON_LOGS",
    help="Output logs in JSON format",
)
@click.option(
    "--include-failures/--exclude-failures",
    "include_failures",
    default=False,
    envvar="INCLUDE_FAILURES",
    show_default=True,
    help="Include/exclude files with previous failures for retry",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform dry run without persisting changes",
)
def main(
    once: bool,
    limit: int,
    poll_seconds: float,
    extensions: str | None,
    max_chars: int | None,
    enable_embedding: bool,
    embedder: str,
    log_level: str,
    log_file: str | None,
    json_logs: bool,
    include_failures: bool,
    dry_run: bool,
) -> None:
    """
    File extraction and embedding worker.
    
    Processes files from the database, extracts text, generates embeddings,
    and persists results. Supports both one-shot and continuous polling modes.
    """
    # Load environment variables
    load_dotenv()
    
    # Configure logging first
    configure_logging(
        level=log_level,
        log_file=log_file,
        console=True,
        json_format=json_logs,
    )
    
    logger = get_logger(__name__)
    
    logger.info("Starting extraction worker CLI")
    
    if dry_run:
        logger.info("Dry run enabled: no database changes will be persisted")
    
    # Parse extensions
    ext_set = None
    if extensions:
        ext_set = set(ext.strip() for ext in extensions.split(",") if ext.strip())
        logger.info(f"Filtering to extensions: {ext_set}")
    
    # Build extractors
    extractors = [
        PDFTextExtractor(),
        TextFileTextExtractor(),
    ]
    logger.info(f"Initialized {len(extractors)} extractors")
    
    # Build embedder (lazy import to avoid loading heavy dependencies on --help)
    embedder_instance = None
    if enable_embedding:
        if embedder == "minilm":
            from embedding.minilm import MiniLMEmbedder
            embedder_instance = MiniLMEmbedder()
            logger.info(f"Initialized MiniLM embedder (dim={embedder_instance.dim})")
        else:
            logger.error(f"Unknown embedder: {embedder}")
            sys.exit(2)
    else:
        logger.info("Embedding disabled")
    
    # Create database session factory
    try:
        engine = get_db_engine()
        session_factory = sessionmaker(bind=engine)
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(2)
    
    # Run worker
    try:
        exit_code = run_worker(
            session_factory=session_factory,
            extractors=extractors,
            embedder=embedder_instance,
            poll_seconds=poll_seconds,
            once=once,
            limit=limit,
            extensions=ext_set,
            max_chars=max_chars,
            enable_embedding=enable_embedding,
            include_failures=include_failures,
            dry_run=dry_run,
        )
        
        logger.info(f"Worker exited with code {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.exception(f"Fatal error in worker: {e}")
        sys.exit(3)


if __name__ == "__main__":
    sys.exit(main())
