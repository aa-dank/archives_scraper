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
- Failures are recorded in the `file_content_failures` table with stage, error, and attempt count
- `FileContent` rows only exist for successful extractions (Option A semantics)
- Failed files are excluded from requeue by default, preventing infinite loops
- Successful processing clears any prior failure record

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
--include-failures / --exclude-failures
                          Include/exclude previously failed files [default: exclude]
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
export INCLUDE_FAILURES=false
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

## Failure Tracking

The system uses a dedicated `file_content_failures` table to track processing failures:

```sql
CREATE TABLE file_content_failures (
    file_hash TEXT PRIMARY KEY,
    stage TEXT NOT NULL CHECK (stage IN ('extract', 'embed')),
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 1,
    last_failed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Semantics:**
- `stage` indicates where failure occurred: `extract` (text extraction) or `embed` (embedding generation)
- `attempts` increments on each retry failure
- On success, the failure record is deleted
- By default, files with failure records are excluded from processing
- Use `--include-failures` to retry previously failed files

**Retry example:**
```bash
python -m cli --include-failures --once --limit 50
```

## Migration from Old System

The old monolithic `main.py` has been preserved as `main_old.py`. The new system provides:
- Better separation of concerns
- Structured logging instead of print statements
- Dedicated failure tracking table (no more sentinel rows)
- CLI argument parsing with environment variable support
- Testable, callable components (no global state)
