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

import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql import func

from db.models import File, FileContent
import logging

logger = logging.getLogger(__name__)


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
    cutoff: datetime | None = None,
) -> list:
    """
    Fetch the next batch of files needing content extraction.
    
    Returns files that either:
    - Have no FileContent row
    - Have FileContent with no updated_at (stale sentinel)
    - Have FileContent updated before cutoff (for reprocessing)
    
    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    extensions : set[str] | None
        If provided, only return files with these extensions (case-insensitive).
    limit : int, default=10
        Maximum number of files to return.
    cutoff : datetime | None
        If provided, include files with content updated before this time.
    
    Returns
    -------
    list
        List of File records needing processing.
    """
    query = (
        session.query(File)
        .options(selectinload(File.locations))
        .outerjoin(FileContent, FileContent.file_hash == File.hash)
    )
    
    # Filter by extensions if provided
    if extensions:
        normalized = [ext.lower().lstrip('.') for ext in extensions]
        query = query.filter(func.lower(File.extension).in_(normalized))
    
    # Filter for files needing processing
    conditions = [
        FileContent.file_hash.is_(None),  # No content row at all
        FileContent.updated_at.is_(None),  # Sentinel row without timestamp
    ]
    
    if cutoff:
        conditions.append(FileContent.updated_at < cutoff)
    
    query = query.filter(or_(*conditions)).order_by(File.id).limit(limit)
    
    files = query.all()
    logger.debug(f"Fetched {len(files)} files needing content extraction")
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
) -> dict:
    """
    Process a single file: extract text, embed, and persist results.
    
    Implements proper failure semantics:
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
        Status values: "ok", "no_extractor", "error"
    """
    start_time = time.time()
    result = {
        "status": "error",
        "chars": 0,
        "duration_ms": 0,
    }
    
    try:
        # Determine extension
        ext = (file_record.extension or "").lower().lstrip('.')
        
        # Select extractor
        extractor = extractors_by_ext.get(ext)
        if not extractor:
            logger.warning(
                f"No extractor for file",
                extra={
                    "file_id": file_record.id,
                    "ext": ext,
                    "path": getattr(file_record, 'path', None),
                }
            )
            # Persist sentinel to prevent infinite requeue
            _persist_failure_sentinel(session, file_record, now_fn)
            result["status"] = "no_extractor"
            result["duration_ms"] = int((time.time() - start_time) * 1000)
            return result
        
        # Get file path from first location
        file_path = None
        if file_record.locations:
            file_path = getattr(file_record.locations[0], 'local_filepath', lambda x: None)(None)
        if not file_path:
            file_path = getattr(file_record, 'path', None)
        
        if not file_path:
            logger.error(
                f"No path available for file",
                extra={"file_id": file_record.id}
            )
            _persist_failure_sentinel(session, file_record, now_fn)
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
        text = extractor(str(file_path))
        
        # Truncate if needed
        if max_chars and len(text) > max_chars:
            logger.warning(
                f"Truncating extracted text",
                extra={
                    "file_id": file_record.id,
                    "original_chars": len(text),
                    "max_chars": max_chars,
                }
            )
            text = text[:max_chars]
        
        result["chars"] = len(text)
        
        # Generate embedding if enabled
        embedding_vector = None
        if enable_embedding and text.strip():
            logger.debug(
                f"Generating embedding for file",
                extra={"file_id": file_record.id, "chars": len(text)}
            )
            embeddings = embedder.encode([text])
            if embeddings and len(embeddings) > 0:
                embedding_vector = embeddings[0]
        
        # Upsert FileContent
        content = file_record.content
        if content is None:
            content = FileContent(file_hash=file_record.hash)
            file_record.content = content
        
        content.source_text = text
        content.text_length = len(text)
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
        
        session.add(content)
        session.commit()
        
        logger.info(
            f"Successfully processed file",
            extra={
                "file_id": file_record.id,
                "chars": len(text),
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
            }
        )
        
        # Persist failure marker in new transaction
        try:
            _persist_failure_sentinel(session, file_record, now_fn)
        except Exception as persist_error:
            logger.error(
                f"Failed to persist error sentinel",
                extra={
                    "file_id": file_record.id,
                    "error": str(persist_error),
                }
            )
        
        result["status"] = "error"
        result["duration_ms"] = int((time.time() - start_time) * 1000)
        return result


def _persist_failure_sentinel(
    session: Session,
    file_record: Any,
    now_fn: Callable[[], datetime],
) -> None:
    """
    Persist an empty FileContent row to mark a file as processed but failed.
    
    This prevents infinite requeue loops for files that cannot be processed.
    
    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    file_record : Any
        File model instance.
    now_fn : Callable[[], datetime]
        Function returning current UTC datetime.
    """
    content = file_record.content
    if content is None:
        content = FileContent(file_hash=file_record.hash)
        file_record.content = content
    
    # Empty text with updated timestamp serves as sentinel
    content.source_text = ""
    content.text_length = 0
    content.updated_at = now_fn()
    
    session.add(content)
    session.commit()
    
    logger.debug(
        f"Persisted failure sentinel",
        extra={"file_id": file_record.id}
    )


def run_worker(
    *,
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
    
    logger.info(
        f"Worker starting",
        extra={
            "once": once,
            "poll_seconds": poll_seconds,
            "limit": limit,
            "extensions": list(extensions) if extensions else None,
            "enable_embedding": enable_embedding,
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
