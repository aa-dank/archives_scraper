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

# Load environment variables before importing project modules, because several
# extractors read their subprocess limits during module initialization.
load_dotenv()

from sqlalchemy.orm import sessionmaker

from db.db import get_db_engine
from logging_configuration import configure_logging, get_logger
from text_extraction.basic_extraction import TextFileTextExtractor
from text_extraction.image_extraction import ImageTextExtractor
from text_extraction.office_doc_extraction import (
    PresentationTextExtractor,
    SpreadsheetTextExtractor,
    WordFileTextExtractor,
)
from text_extraction.pdf_extraction import PDFTextExtractor
from text_extraction.web_extraction import EmailTextExtractor, HtmlTextExtractor
from worker import run_worker


@click.command()
@click.option(
    "--limit",
    type=int,
    envvar="LIMIT",
    help="Total files to process before exiting. If omitted, run continuously.",
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
    "--max-runtime-seconds",
    type=float,
    envvar="MAX_RUNTIME_SECONDS",
    help="Maximum runtime in seconds before exiting cleanly.",
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
    "--no-date-extract",
    "enable_date_extraction",
    is_flag=True,
    flag_value=False,
    default=True,
    envvar="ENABLE_DATE_EXTRACT",
    help="Disable date mention extraction (extraction runs by default)",
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
    "--randomize/--no-randomize",
    "randomize",
    default=False,
    envvar="RANDOMIZE",
    show_default=True,
    help="Randomize database file selection order for scraping",
)
@click.option(
    "--hashes",
    type=str,
    envvar="TARGET_HASHES",
    help="Comma-separated file hashes to process once, including already-processed files",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform dry run without persisting changes",
)
def main(
    limit: int | None,
    poll_seconds: float,
    max_runtime_seconds: float | None,
    extensions: str | None,
    max_chars: int | None,
    enable_embedding: bool,
    enable_date_extraction: bool,
    embedder: str,
    log_level: str,
    log_file: str | None,
    json_logs: bool,
    include_failures: bool,
    randomize: bool,
    hashes: str | None,
    dry_run: bool,
) -> None:
    """
    File extraction and embedding worker.
    
    Processes files from the database, extracts text, generates embeddings,
    and persists results. Runs continuously unless a limit is provided.
    """
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

    target_hashes = None
    if hashes is not None:
        target_hashes = {file_hash.strip() for file_hash in hashes.split(",") if file_hash.strip()}
        if not target_hashes:
            raise click.UsageError("--hashes must include at least one non-empty file hash")
        logger.info("Targeted mode enabled", extra={"target_hash_count": len(target_hashes)})
    
    # Build extractors
    extractors = [
        PDFTextExtractor(),
        ImageTextExtractor(),
        WordFileTextExtractor(),
        PresentationTextExtractor(),
        SpreadsheetTextExtractor(),
        HtmlTextExtractor(),
        EmailTextExtractor(),
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
            limit=limit,
            max_runtime_seconds=max_runtime_seconds,
            extensions=ext_set,
            max_chars=max_chars,
            enable_embedding=enable_embedding,
            enable_date_extraction=enable_date_extraction,
            include_failures=include_failures,
            randomize=randomize,
            target_hashes=target_hashes,
            dry_run=dry_run,
        )
        
        logger.info(f"Worker exited with code {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.exception(f"Fatal error in worker: {e}")
        sys.exit(3)


if __name__ == "__main__":
    sys.exit(main())
