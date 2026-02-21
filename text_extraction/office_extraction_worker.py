import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from .basic_extraction import TextExtractionError
from .extraction_utils import SubprocessUtils, validate_file
from .office_doc_extraction import (
    OFFICE_WORKER_MEM_MB,
    OFFICE_WORKER_MODE_ENV,
    OFFICE_WORKER_TIMEOUT_S,
    OfficeConverter,
    PresentationTextExtractor,
    SpreadsheetTextExtractor,
    WordFileTextExtractor,
)

logger = logging.getLogger(__name__)

WORD_EXTENSIONS = {"doc", "docx", "docm"}
PRESENTATION_EXTENSIONS = {"ppt", "pps", "pptx", "pptm", "ppsx"}
SPREADSHEET_EXTENSIONS = {"xls", "xlsx", "xlsm"}


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr, format="%(levelname)s: %(message)s")


def _parse_config(config_json_arg: str | None) -> dict:
    if not config_json_arg:
        return {}

    config = json.loads(config_json_arg)
    if not isinstance(config, dict):
        raise ValueError("--config-json must decode to a JSON object")
    return config


def _int_config(config: dict, *keys: str, default: int) -> int:
    node = config
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    if node is None:
        return default
    try:
        return int(node)
    except (TypeError, ValueError):
        return default


def _bool_config(config: dict, *keys: str, default: bool) -> bool:
    node = config
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    if node is None:
        return default
    return bool(node)


def _reason_from_exception(exc: BaseException) -> tuple[str, bool]:
    message = str(exc).lower()
    if "timed out" in message:
        return "timeout", True
    if "executable not found" in message or "soffice" in message and "not found" in message:
        return "converter_missing", False
    if "unsupported extension" in message:
        return "unsupported_extension", False
    if "conversion" in message:
        return "conversion_failed", False
    if isinstance(exc, MemoryError):
        return "oom_killed", False
    return "parse_failed", False


def _build_word_extractor(config: dict) -> WordFileTextExtractor:
    soffice_path = str(config.get("soffice_path", "soffice"))
    max_output_chars = _int_config(config, "max_output_chars", default=5_000_000)
    return WordFileTextExtractor(
        converter=OfficeConverter(soffice_path=soffice_path),
        max_output_chars=max_output_chars,
    )


def _build_presentation_extractor(config: dict) -> PresentationTextExtractor:
    soffice_path = str(config.get("soffice_path", "soffice"))
    include_notes = _bool_config(config, "include_ppt_notes", default=True)
    max_slides = _int_config(config, "ppt", "max_slides", default=500)
    max_chars_per_slide = _int_config(config, "ppt", "max_chars_per_slide", default=50_000)
    return PresentationTextExtractor(
        include_notes=include_notes,
        max_slides=max_slides,
        max_chars_per_slide=max_chars_per_slide,
        converter=OfficeConverter(soffice_path=soffice_path, default_timeout_s=120),
    )


def _build_spreadsheet_extractor(config: dict) -> SpreadsheetTextExtractor:
    soffice_path = str(config.get("soffice_path", "soffice"))
    max_sheets = _int_config(config, "xlsx", "max_sheets", default=20)
    max_rows_per_sheet = _int_config(config, "xlsx", "max_rows_per_sheet", default=5_000)
    max_cols_per_sheet = _int_config(config, "xlsx", "max_cols_per_sheet", default=200)
    max_total_cells = _int_config(config, "xlsx", "max_total_cells", default=500_000)

    return SpreadsheetTextExtractor(
        max_sheets=max_sheets,
        max_rows_per_sheet=max_rows_per_sheet,
        max_cols_per_sheet=max_cols_per_sheet,
        max_total_cells=max_total_cells,
        converter=OfficeConverter(soffice_path=soffice_path),
    )


def _extract_text(input_path: Path, config: dict) -> str:
    ext = input_path.suffix.lower().lstrip(".")

    if ext in WORD_EXTENSIONS:
        return _build_word_extractor(config)(str(input_path))
    if ext in PRESENTATION_EXTENSIONS:
        return _build_presentation_extractor(config)(str(input_path))
    if ext in SPREADSHEET_EXTENSIONS:
        return _build_spreadsheet_extractor(config)(str(input_path))

    raise TextExtractionError(f"Unsupported extension for office worker: {ext}")


def _emit_json(payload: dict) -> int:
    print(json.dumps(payload), end="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run office extraction in an isolated subprocess")
    parser.add_argument("--input", required=True, help="Path to office document")
    parser.add_argument("--config-json", default="{}", help="Worker config JSON")
    parser.add_argument("--timeout-s", type=int, default=OFFICE_WORKER_TIMEOUT_S, help="Worker timeout in seconds")
    parser.add_argument("--mem-mb", type=int, default=OFFICE_WORKER_MEM_MB, help="Worker memory cap in MB")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging and preserve temp files")

    args = parser.parse_args()

    _configure_logging(args.debug)

    if args.debug:
        os.environ["DEBUG_KEEP_TEMPS"] = "1"

    os.environ[OFFICE_WORKER_MODE_ENV] = "1"

    started = time.time()
    metadata = {
        "worker_mode": True,
        "worker_mem_mb": args.mem_mb,
        "worker_timeout_s": args.timeout_s,
        "duration_ms": 0,
    }

    try:
        SubprocessUtils.apply_memory_limit(args.mem_mb, stderr=sys.stderr)

        input_path = validate_file(args.input)
        config = _parse_config(args.config_json)

        logger.info("Office worker start: path=%s ext=%s", input_path, input_path.suffix.lower())
        text = _extract_text(input_path, config)

        metadata["duration_ms"] = int((time.time() - started) * 1000)
        return _emit_json(
            {
                "ok": True,
                "text": text,
                "metadata": metadata,
            }
        )
    except Exception as exc:
        reason, retryable = _reason_from_exception(exc)
        metadata["duration_ms"] = int((time.time() - started) * 1000)
        logger.exception("Office worker failed: %s", exc)
        return _emit_json(
            {
                "ok": False,
                "metadata": metadata,
                "error": {
                    "reason": reason,
                    "details": {
                        "message": str(exc),
                        "exception_type": exc.__class__.__name__,
                    },
                    "retryable": retryable,
                },
            }
        )


if __name__ == "__main__":
    raise SystemExit(main())
