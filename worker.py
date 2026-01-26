# worker.py

"""
Pure execution engine for file extraction and embedding.

This module provides the core worker logic for:
- Fetching unprocessed files from the database
- Extracting text using registered extractors
- Embedding text using configured embedders
- Persisting results with proper failure handling

No CLI parsing or global state - callable from anywhere.
"""
import os
import time
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from text_extraction.extraction_utils import (
    common_char_replacements,
    strip_diacritics,
    normalize_unicode,
    normalize_whitespace
)

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql import func

from db.models import File, FileContent, FileContentFailure
import logging

logger = logging.getLogger(__name__)

# Stage constants for file_content_failures
STAGE_EXTRACT = "extract"
STAGE_EMBED = "embed"

def assemble_file_server_filepath(base_mount: str,
                                  server_dir: str,
                                  filename: str = None) -> Path:
    r"""
    Join a server-relative path + filename onto a machine-specific
    mount-point.

    Parameters
    ----------
    base_mount : str
        The local mount of the records share, e.g.
        r"N:\PPDO\Records"  (Windows)  or  "/mnt/n/PPDO/Records" (Linux).
    server_dir : str
        The value from file_locations.file_server_directories
        (always stored with forward-slashes).
    filename   : str
        file_locations.filename

    Returns
    -------
    pathlib.Path  – ready for open(), exists(), etc.
    """
    # 1) Treat the DB field as a *POSIX* path (it always uses “/”)
    rel_parts = PurePosixPath(server_dir).parts     # -> tuple of segments

    # 2) Let Path figure out the separator style of this machine
    full_path = Path(base_mount).joinpath(*rel_parts)
    if filename:
        full_path = full_path / filename
    
    return full_path

def utcnow() -> datetime:
    """Return current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


def build_extractor_registry(extractors: list) -> dict[str, Any]:
    """
    Build a mapping of file extension to extractor.
    
    Extensions are normalized to lowercase without leading dots.
    Conflict resolution is last-wins (last extractor in list takes precedence).
    
    Parameters
    ----------
    extractors : list
        List of extractor instances, each with a `file_extensions` attribute.
    
    Returns
    -------
    dict[str, Any]
        Mapping of normalized extension (e.g., "pdf") to extractor instance.
    """
    registry = {}
    for extractor in extractors:
        if not hasattr(extractor, 'file_extensions'):
            logger.warning(f"Extractor {extractor} missing file_extensions attribute, skipping")
            continue
        
        for ext in extractor.file_extensions:
            # Normalize: lowercase, strip leading dot
            normalized = ext.lower().lstrip('.')
            registry[normalized] = extractor
            logger.debug(f"Registered extractor {extractor.__class__.__name__} for extension '{normalized}'")
    
    logger.info(f"Built extractor registry with {len(registry)} extensions")
    return registry


def next_files_needing_content(
    session: Session,
    *,
    extensions: set[str] | None = None,
    limit: int = 10,
    include_failures: bool = False,
) -> list:
    """
    Fetch the next batch of files needing content extraction.
    
    Returns files that have no FileContent row (Option A semantics).
    By default, excludes files with existing failure records to prevent
    infinite requeue loops.
    
    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    extensions : set[str] | None
        If provided, only return files with these extensions (case-insensitive).
    limit : int, default=10
        Maximum number of files to return.
    include_failures : bool, default=False
        If True, include files that have failure records (for retry).
        If False (default), exclude files with any failure record.
    
    Returns
    -------
    list
        List of File records needing processing.
    """
    query = (
        session.query(File)
        .options(selectinload(File.locations))
        .outerjoin(FileContent, FileContent.file_hash == File.hash)
        .outerjoin(FileContentFailure, FileContentFailure.file_hash == File.hash)
    )
    
    # Filter by extensions if provided
    if extensions is not None:
        normalized = [ext.lower().lstrip('.') for ext in extensions]
        query = query.filter(func.lower(File.extension).in_(normalized))
    
    # Base condition: no successful FileContent row (Option A)
    query = query.filter(FileContent.file_hash.is_(None))
    
    # Apply failure filtering
    if not include_failures:
        # Exclude files that have a failure record
        query = query.filter(FileContentFailure.file_hash.is_(None))
    
    query = query.order_by(File.id).limit(limit)
    
    files = query.all()
    logger.debug(f"Fetched {len(files)} files needing content extraction (include_failures={include_failures})")
    return files


def process_one_file(
    session: Session,
    *,
    extractors_by_ext: dict[str, Any],
    embedder: Any,
    file_record: Any,
    now_fn: Callable[[], datetime] = utcnow,
    max_chars: int | None = None,
    enable_embedding: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Process a single file: extract text, embed, and persist results.
    
    Implements proper failure semantics:
    dry_run: bool = False,
    - No extractor: persist empty FileContent sentinel to prevent requeue
    - Exception: rollback and persist failure marker in new transaction
    
    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    extractors_by_ext : dict[str, Any]
        Mapping from extension to extractor instance.
    embedder : Any
        Embedder instance with encode(Sequence[str]) -> list[np.ndarray] method.
    file_record : Any
        File model instance to process.
    now_fn : Callable[[], datetime]
        Function returning current UTC datetime (for testing).
    max_chars : int | None
        If set, truncate extracted text to this length.
    enable_embedding : bool
        Whether to generate embeddings (default True).
    
    Returns
    -------
    dict
        Status information with keys: status, chars, duration_ms
    dry_run : bool
        If True, do not persist changes to database.
        Status values: "ok", "no_extractor", "error"
    """
    start_time = time.time()
    result = {
        "status": "error",
        "chars": 0,
        "duration_ms": 0,
    }
    current_stage = STAGE_EXTRACT
    
    try:
        # Determine extension
        ext = (file_record.extension or "").lower().lstrip('.')
        
        # Select extractor
        extractor = extractors_by_ext.get(ext)
        if not extractor:
            error_msg = f"no extractor for ext={ext}"
            logger.warning(
                f"No extractor for file",
                extra={
                    "file_id": file_record.id,
                    "ext": ext,
                    "path": getattr(file_record, 'path', None),
                    "stage": STAGE_EXTRACT,
                }
            )
            if not dry_run:
                # Record failure to prevent infinite requeue
                _upsert_failure(session, file_record.hash, STAGE_EXTRACT, error_msg, now_fn)
            result["status"] = "no_extractor"
            result["duration_ms"] = int((time.time() - start_time) * 1000)
            return result
        
        # Get file path from first location
        file_path = None
        if file_record.locations:
            record_location_directories = file_record.locations[0].file_server_directories
            record_filename = file_record.locations[0].filename
            file_path = assemble_file_server_filepath(
                base_mount=os.environ.get("FILE_SERVER_MOUNT", ""),
                server_dir=record_location_directories,
                filename=record_filename,
            )
        
        if not file_path:
            error_msg = "no path available"
            logger.error(
                f"No path available for file",
                extra={"file_id": file_record.id, "stage": STAGE_EXTRACT}
            )
            if not dry_run:
                _upsert_failure(session, file_record.hash, STAGE_EXTRACT, error_msg, now_fn)
            result["status"] = "error"
            result["duration_ms"] = int((time.time() - start_time) * 1000)
            return result
        
        # Extract text
        logger.info(
            f"Extracting text from file",
            extra={
                "file_id": file_record.id,
                "path": str(file_path),
                "ext": ext,
            }
        )
        
        extracted_text = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_fp = os.path.join(temp_dir, os.path.basename(str(file_path)))
            shutil.copyfile(str(file_path), temp_fp)
            extracted_text = extractor(temp_fp)

        if extracted_text:
            extracted_text = common_char_replacements(extracted_text)
            extracted_text = strip_diacritics(extracted_text)
            extracted_text = normalize_unicode(extracted_text)
            extracted_text = normalize_whitespace(extracted_text)
        
        # Truncate if needed
        if max_chars and len(extracted_text) > max_chars:
            logger.warning(
                f"Truncating extracted text",
                extra={
                    "file_id": file_record.id,
                    "original_chars": len(extracted_text),
                    "max_chars": max_chars,
                }
            )
            
            # truncate the text by cutting at last newline/space before max_chars
            truncated = extracted_text[:max_chars]
            last_break = max(truncated.rfind("\n"), truncated.rfind(" "))
            if last_break > max_chars * 0.8:
                truncated = truncated[:last_break]
            extracted_text = truncated
        
        result["chars"] = len(extracted_text)
        
        # Generate embedding if enabled
        current_stage = STAGE_EMBED
        embedding_vector = None
        if enable_embedding and extracted_text.strip():
            logger.debug(
                f"Generating embedding for file",
                extra={"file_id": file_record.id, "chars": len(extracted_text)}
            )
            embeddings = embedder.encode([extracted_text])
            if embeddings and len(embeddings) > 0:
                embedding_vector = embeddings[0]
        
        # Upsert FileContent
        content = file_record.content
        if content is None:
            content = FileContent(file_hash=file_record.hash)
            file_record.content = content
        
        content.source_text = extracted_text
        content.text_length = len(extracted_text)
        content.updated_at = now_fn()
        
        if embedding_vector is not None:
            # Determine embedder model name
            model_name = getattr(embedder, 'model_name', 'unknown')
            
            # Map to appropriate column based on dimension
            if hasattr(embedding_vector, 'shape'):
                dim = embedding_vector.shape[0] if len(embedding_vector.shape) > 0 else len(embedding_vector)
            else:
                dim = len(embedding_vector)
            
            if dim == 384:  # MiniLM dimension
                content.minilm_emb = embedding_vector
                content.minilm_model = model_name
            elif dim == 768:  # MPNet dimension
                content.mpnet_emb = embedding_vector
                content.mpnet_model = model_name
            else:
                logger.warning(
                    f"Unknown embedding dimension",
                    extra={"file_id": file_record.id, "dimension": dim}
                )
        if not dry_run:
            session.add(content)
            session.commit()

            # Clear any existing failure record on success
            _clear_failure(session, file_record.hash)
        else:
            session.rollback()
        
        logger.info(
            f"Successfully processed file",
            extra={
                "file_id": file_record.id,
                "chars": len(extracted_text),
                "status": "ok",
            }
        )
        
        result["status"] = "ok"
        result["duration_ms"] = int((time.time() - start_time) * 1000)
        return result
        
    except Exception as e:
        # Rollback current transaction
        session.rollback()
        
        logger.exception(
            f"Error processing file",
            extra={
                "file_id": file_record.id,
                "error": str(e),
                "stage": current_stage,
            }
        )
        
        # Record failure in new transaction
        try:
            _upsert_failure(session, file_record.hash, current_stage, str(e)[:500], now_fn)
        except Exception as persist_error:
            logger.error(
                f"Failed to record failure",
                extra={
                    "file_id": file_record.id,
                    "error": str(persist_error),
                }
            )
        
        result["status"] = "error"
        result["duration_ms"] = int((time.time() - start_time) * 1000)
        return result


def _upsert_failure(
    session: Session,
    file_hash: str,
    stage: str,
    error: str,
    now_fn: Callable[[], datetime] = utcnow,
) -> None:
    """
    Insert or update a failure record in file_content_failures.
    
    On conflict (existing failure for this file_hash), increments attempts
    and updates the stage, error message, and timestamp.
    
    dry_run: bool = False,
    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    file_hash : str
        Hash of the file that failed.
    stage : str
        Processing stage that failed (STAGE_EXTRACT or STAGE_EMBED).
    error : str
        Human-readable error message.
    now_fn : Callable[[], datetime]
        Function returning current UTC datetime.
    """
    upsert_sql = text("""
        INSERT INTO file_content_failures (file_hash, stage, error, attempts, last_failed_at)
        VALUES (:file_hash, :stage, :error, 1, :now)
        ON CONFLICT (file_hash)
        DO UPDATE SET
            stage = EXCLUDED.stage,
            error = EXCLUDED.error,
            attempts = file_content_failures.attempts + 1,
            last_failed_at = EXCLUDED.last_failed_at
    """)
    
    session.execute(
        upsert_sql,
        {"file_hash": file_hash, "stage": stage, "error": error, "now": now_fn()}
    )
    session.commit()
    
    logger.debug(
        f"Recorded failure",
        extra={"file_hash": file_hash, "stage": stage}
    )


def _clear_failure(session: Session, file_hash: str) -> None:
    """
    Delete any existing failure record for a file upon successful processing.
    
    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    file_hash : str
        Hash of the successfully processed file.
    """
    result = session.query(FileContentFailure).filter(
        FileContentFailure.file_hash == file_hash
    ).delete()
    
    if result > 0:
        session.commit()
        logger.debug(
            f"Cleared failure record on success",
            extra={"file_hash": file_hash}
        )


def run_worker(
    *,
    dry_run: bool = False,
    session_factory: Callable,
    extractors: list,
    embedder: Any,
    poll_seconds: float = 5.0,
    once: bool = False,
    limit: int | None = None,
    extensions: set[str] | None = None,
    max_chars: int | None = None,
    backoff_seconds: float = 30.0,
    enable_embedding: bool = True,
    include_failures: bool = False,
) -> int:
    """
    Main worker execution loop.
    
    Continuously fetches and processes files until stopped or no more work.
    
    Parameters
    ----------
    session_factory : Callable
        Factory function returning SQLAlchemy sessions (e.g., sessionmaker).
    extractors : list
        List of extractor instances.
    embedder : Any
        Embedder instance.
    poll_seconds : float, default=5.0
        Seconds to sleep between polling when no work found.
    once : bool, default=False
        If True, exit after one pass regardless of results.
    limit : int | None
        Maximum files to process per batch.
    extensions : set[str] | None
        If provided, only process files with these extensions.
    max_chars : int | None
        If set, truncate extracted text to this length.
    backoff_seconds : float, default=30.0
        Seconds to sleep when no work found before next poll.
    enable_embedding : bool, default=True
        Whether to generate embeddings.
    include_failures : bool, default=False
        If True, include files with failure records for retry.
        If False (default), exclude files that have previously failed.
    
    Returns
    -------
    int
        Exit code: 0 (clean), 2 (config error), 3 (runtime failure)
    """
    # Validate configuration
    if not extractors:
        logger.error("No extractors provided")
        return 2
    
    if enable_embedding and not embedder:
        logger.error("Embedding enabled but no embedder provided")
        return 2
    
    # Build extractor registry
    registry = build_extractor_registry(extractors)
    if not registry:
        logger.error("Failed to build extractor registry")
        return 2

    # Restrict extensions to those supported by extractors
    supported_extensions = set(registry.keys())
    if extensions:
        extensions = extensions.intersection(supported_extensions)
    else:
        extensions = supported_extensions
    
    logger.info(
        f"Worker starting",
        extra={
            "once": once,
            "poll_seconds": poll_seconds,
            "limit": limit,
            "extensions": list(extensions) if extensions else None,
            "enable_embedding": enable_embedding,
            "include_failures": include_failures,
        }
    )
    
    batch_limit = limit or 10
    total_processed = 0
    
    try:
        while True:
            with session_factory() as session:
                # Fetch next batch
                files = next_files_needing_content(
                    session,
                    extensions=extensions,
                    limit=batch_limit,
                    include_failures=include_failures,
                )
                
                if not files:
                    logger.info("No files needing processing")
                    if once:
                        logger.info(f"Exiting after processing {total_processed} files (once mode)")
                        return 0
                    
                    logger.debug(f"Sleeping {backoff_seconds}s before next poll")
                    time.sleep(backoff_seconds)
                    continue
                
                logger.info(f"Processing batch of {len(files)} files")
                
                # Process each file
                batch_results = {"ok": 0, "no_extractor": 0, "error": 0}
                for file_record in files:
                    result = process_one_file(
                        session,
                        extractors_by_ext=registry,
                        embedder=embedder,
                        file_record=file_record,
                        max_chars=max_chars,
                        enable_embedding=enable_embedding,
                        dry_run=dry_run,
                    )
                    batch_results[result["status"]] += 1
                    total_processed += 1
                
                logger.info(
                    f"Batch complete",
                    extra={
                        "processed": len(files),
                        "ok": batch_results["ok"],
                        "no_extractor": batch_results["no_extractor"],
                        "errors": batch_results["error"],
                    }
                )
                
                if once:
                    logger.info(f"Exiting after processing {total_processed} files (once mode)")
                    return 0
                
                # Brief sleep before next batch
                if poll_seconds > 0:
                    time.sleep(poll_seconds)
    
    except KeyboardInterrupt:
        logger.info(f"Worker interrupted, processed {total_processed} files")
        return 0
    
    except Exception as e:
        logger.exception(f"Unexpected worker failure: {e}")
        return 3
