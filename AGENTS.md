# Archives Scraper

## Build, test, and run commands

Use `python3` in shell commands unless a virtualenv has already provided a `python` shim.

Install the package in editable mode:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Smoke-check the CLI entrypoint and options:

```bash
python3 cli.py --help
python3 -m cli --help
```

Run the worker through the CLI, not by invoking `worker.py` directly:

```bash
python3 cli.py --limit 10
python3 cli.py --extensions pdf --log-level DEBUG --limit 10
python3 cli.py --no-embed --limit 10
python3 cli.py --include-failures --limit 50
python3 cli.py --poll-seconds 5.0
```

There is no repo-configured lint command in `pyproject.toml`.

This project does not have an automated test suite; validate changes with targeted CLI smoke checks instead.

## Embedding Dependency Notes

- Extraction-only operation (`--no-embed` or `ENABLE_EMBEDDING=false`) does not require `sentence-transformers`, `torch`, or CUDA runtime libraries at execution time.
- Embedding-enabled operation (`--embed`) uses `embedding/minilm.py`, which imports `sentence-transformers` and transitively depends on `torch`.
- CUDA is not functionally required for this project; CPU embeddings are valid.
- On Linux, newer `torch` resolutions may pull NVIDIA/CUDA transitive packages. Treat those as an environment-resolution artifact, not a core requirement for extraction workflows.
- When changing dependency management, preserve a clear extraction-only path that avoids forcing GPU stacks for users who do not need embeddings.

## Reference docs

Link to these docs rather than duplicating long explanations in prompts or generated notes:

- `README.md` for full CLI options, interfaces, and examples.
- `development/failure_table_integration_spec.md` for failure persistence behavior.
- `development/ocr_subprocess_spec.md` and `development/office_extraction_worker_spec.md` for subprocess routing details.
- `/home/projects/business_services_db/reference/ARCHIVES_DB_AND_FILE_SERVER_REFERENCE.md` for upstream database and file-server context used by this project.

## High-level architecture

- `cli.py` is a thin operational wrapper. It loads `.env`, configures logging, instantiates the extractor list and embedder, creates the SQLAlchemy session factory, and then calls `run_worker(...)`.
- `worker.py` is the core pipeline. It queries `files`/`file_locations` for records without a successful `file_contents` row, resolves server-relative paths using `FILE_SERVER_MOUNT`, copies each source file into a temporary working directory, runs the extractor, normalizes the extracted text, optionally extracts date mentions, optionally generates embeddings, and persists normalized output.
- Failure handling is first-class. Failures are written to `file_content_failures`, and successful runs clear any prior failure row. By default the worker excludes failed files from requeue; `--include-failures` retries them.
- Persistence is hash-centered. `FileContent` and `FileContentFailure` use `files.hash` as the key, so extraction/embedding work is tracked per file hash rather than per location row.
- Extractors live under `text_extraction/`. `PDFTextExtractor`, `ImageTextExtractor`, `HtmlTextExtractor`, `EmailTextExtractor`, `WordFileTextExtractor`, `PresentationTextExtractor`, `SpreadsheetTextExtractor`, and `TextFileTextExtractor` are composed in `cli.py` and registered by file extension in `worker.py`.
- Large or fragile formats are isolated in subprocesses instead of always running in-process. Office extraction can route through `text_extraction.office_extraction_worker`, image OCR can route through `text_extraction.image_extraction_worker`, and legacy Office formats use headless LibreOffice conversion before parsing.
- Embedding models are thin wrappers under `embedding/`. The current CLI-supported embedder is `MiniLMEmbedder`, which writes 384-dimension vectors into `file_contents.minilm_emb`.

## Key conventions

- Extractors declare `file_extensions` as lowercase strings without leading dots. `build_extractor_registry()` normalizes extensions the same way, and registry conflicts are last-wins based on extractor order in `cli.py`.
- A file is considered successfully processed only when a `FileContent` row is written. Missing extractors, extraction failures, max-char violations, and embedding failures are represented through `file_content_failures` instead of success-shaped sentinel rows.
- The worker always copies the source document to a temp path before calling an extractor. Extractors should operate on the provided temp file and should not mutate the mounted source file.
- Failure messages are normalized before persistence. `format_failure_error()` rewrites temp-path references back to the source path and appends `source_path`, `server_dir`, and `filename` context, so error text should remain meaningful after temp directories are cleaned up.
- Text normalization is centralized in the worker after extraction: common character replacements, diacritic stripping, Unicode normalization, and whitespace normalization happen before persistence and embedding.
- Date mention extraction is enabled by default in the CLI. Disabling it uses `--no-date-extract`; the option name is inverted because extraction is on unless explicitly turned off.
- `file_locations.file_server_directories` is stored as a POSIX-style relative path from the server. Path assembly should go through `assemble_file_server_filepath(...)` or `FileLocation.local_filepath(...)` instead of manual string concatenation.
- Structured logging is expected. Log records commonly use `extra={...}` fields such as `file_id`, `path`, `ext`, `chars`, and `duration_ms`, and `logging_configuration.py` supports both concise console logs and JSON logs.
- Office extractors have built-in size and format thresholds that switch work into subprocess mode. Preserve that behavior when changing Office parsing logic rather than pulling everything back into the main worker process.
- Keep CLI orchestration concerns in `cli.py` and processing semantics in `worker.py`; avoid blending command parsing with pipeline behavior.
