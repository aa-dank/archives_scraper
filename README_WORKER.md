# Extraction Worker

This document describes the new modular extraction worker system for processing files, extracting text, and generating embeddings.

## Architecture

The system consists of three main components:

### 1. `logging_configuration.py`
Centralized logging configuration shared by all components.
- Idempotent configuration (safe to call multiple times)
- Supports console, file (rotating), and JSON logging
- Structured logging with `extra={}` context fields

### 2. `worker.py`
Pure execution engine with no CLI dependencies.
- `build_extractor_registry()` - Maps file extensions to extractors
- `next_files_needing_content()` - Batch query for unprocessed files
- `process_one_file()` - Extract text and embed single file
- `run_worker()` - Main execution loop with configurable polling

**Failure Handling:**
- No extractor: Persists empty FileContent sentinel to prevent requeue
- Exception: Rollbacks transaction, persists failure marker in new transaction
- Never allows infinite requeue loops

### 3. `cli.py`
Click-based CLI wrapper around `run_worker()`.
- All major options support environment variables
- Lazy loading of heavy dependencies for fast `--help`

## Usage

### Basic Usage

Process files once and exit:
```bash
python -m cli --once --limit 10
```

Run continuously with polling:
```bash
python -m cli --poll-seconds 5.0
```

### Command-Line Options

```
--once                    Process one batch then exit
--limit INTEGER           Maximum files per batch [default: 10]
--poll-seconds FLOAT      Seconds between polls [default: 5.0]
--extensions TEXT         Comma-separated extensions (e.g., "pdf,txt")
--max-chars INTEGER       Truncate extracted text to N characters
--embed / --no-embed      Enable/disable embedding [default: embed]
--embedder [minilm]       Embedder model to use [default: minilm]
--log-level [debug|info|warning|error]  Logging level [default: INFO]
--log-file PATH           Path to log file
--json-logs               Output logs in JSON format
--dry-run                 Dry run mode (not yet implemented)
```

### Environment Variables

All major options can be set via environment variables:

```bash
export ONCE=true
export LIMIT=50
export POLL_SECONDS=10.0
export EXTENSIONS="pdf,txt,md"
export LOG_LEVEL=DEBUG
export LOG_FILE=/var/log/worker.log
export JSON_LOGS=true
export ENABLE_EMBEDDING=false
```

### Examples

**Extract PDFs only, with debug logging:**
```bash
python -m cli --extensions pdf --log-level debug --once
```

**Continuous mode with file logging:**
```bash
python -m cli --log-file worker.log --poll-seconds 30
```

**Disable embedding, process text files:**
```bash
python -m cli --no-embed --extensions "txt,md,log"
```

**AWS/Production mode with environment variables:**
```bash
export ONCE=true
export LIMIT=100
export LOG_LEVEL=INFO
export LOG_FILE=/var/log/extraction-worker.log
export JSON_LOGS=true
python -m cli
```

## Exit Codes

- `0` - Clean exit (completed successfully or interrupted)
- `2` - Configuration error (no extractors, missing embedder, DB connection failed)
- `3` - Unexpected runtime failure

## Extractor Interface

All extractors must implement:
```python
class MyExtractor:
    # Lowercase extensions without leading dots
    file_extensions = ['pdf', 'docx']
    
    def __call__(self, path: str) -> str:
        # Extract and return text
        pass
```

## Embedder Interface

All embedders must implement:
```python
class MyEmbedder:
    def encode(self, texts: Sequence[str]) -> list[np.ndarray]:
        # Return list of embedding vectors
        pass
```

## Database Requirements

The worker requires the following environment variables for database connection:
- `PROJECT_DB_USERNAME`
- `PROJECT_DB_PASSWORD`
- `PROJECT_DB_HOST`
- `PROJECT_DB_PORT`
- `PROJECT_DB_NAME`

These are typically loaded from a `.env` file in the project root.

## Logging Conventions

Use structured logging with context:

```python
logger.info(
    "Processing file",
    extra={
        "file_id": 123,
        "path": "/path/to/file.pdf",
        "ext": "pdf",
        "chars": 5000,
        "duration_ms": 250,
    }
)
```

This enables JSON logging and easier debugging without code changes.

## Future Enhancements

As noted in the spec, the current system uses empty `FileContent` rows as failure sentinels. The recommended future improvement is a dedicated failure table:

```sql
CREATE TABLE file_content_failures (
    file_id INT,
    stage TEXT,
    error TEXT,
    attempts INT,
    updated_at TIMESTAMP
);
```

This would allow distinguishing between legitimate empty files and processing failures.

## Migration from Old System

The old monolithic `main.py` has been preserved as `main_old.py`. The new system provides:
- Better separation of concerns
- Structured logging instead of print statements
- Proper failure handling to prevent infinite requeues
- CLI argument parsing with environment variable support
- Testable, callable components (no global state)
