import logging
import os
import re
import json
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
import uuid

from pathlib import Path
from typing import List

from .basic_extraction import FileTextExtractor, TextExtractionError
from .extraction_utils import validate_file

logger = logging.getLogger(__name__)

OFFICE_SUBPROCESS_FILESIZE_THRESHOLD_BYTES = int(
    os.getenv("OFFICE_SUBPROCESS_FILESIZE_THRESHOLD_BYTES", str(75 * 1024 * 1024))
)
OFFICE_WORKER_TIMEOUT_S = int(os.getenv("OFFICE_WORKER_TIMEOUT_S", "180"))
OFFICE_WORKER_MODE_ENV = "OFFICE_WORKER_MODE"
OFFICE_WORKER_STDERR_TAIL_MAX_CHARS = 4000
OFFICE_WORKER_MODULE = "text_extraction.office_extraction_worker"

_OFFICE_WORKER_MEM_MB_RAW = os.getenv("OFFICE_WORKER_MEM_MB")
if _OFFICE_WORKER_MEM_MB_RAW is None:
    OFFICE_WORKER_MEM_MB = 2048
else:
    try:
        OFFICE_WORKER_MEM_MB = int(_OFFICE_WORKER_MEM_MB_RAW)
    except ValueError:
        logger.warning("Invalid OFFICE_WORKER_MEM_MB=%s; defaulting to 2048", _OFFICE_WORKER_MEM_MB_RAW)
        OFFICE_WORKER_MEM_MB = 2048

OFFICE_LEGACY_ALWAYS_SUBPROCESS = {"doc", "ppt", "pps", "xls"}
OFFICE_MODERN_SIZE_GATED_SUBPROCESS = {"docx", "docm", "pptx", "pptm", "ppsx", "xlsx", "xlsm"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _tail_text(text: str | None, max_chars: int = OFFICE_WORKER_STDERR_TAIL_MAX_CHARS) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"...[truncated] {cleaned[-max_chars:]}"


def _in_office_worker_mode() -> bool:
    return _env_flag(OFFICE_WORKER_MODE_ENV, default=False)


def _should_route_to_office_subprocess(source: Path, ext: str) -> bool:
    if _in_office_worker_mode():
        return False

    if ext in OFFICE_LEGACY_ALWAYS_SUBPROCESS:
        return True

    if ext in OFFICE_MODERN_SIZE_GATED_SUBPROCESS:
        return source.stat().st_size >= OFFICE_SUBPROCESS_FILESIZE_THRESHOLD_BYTES

    return False


def run_office_worker(
    input_path: Path | str,
    *,
    timeout_s: int = OFFICE_WORKER_TIMEOUT_S,
    mem_mb: int | None = OFFICE_WORKER_MEM_MB,
    config: dict | None = None,
    worker_cmd: list[str] | None = None,
) -> str:
    source = validate_file(str(input_path))
    worker_config = config or {}
    project_root = Path(__file__).resolve().parents[1]

    cmd_override = os.getenv("OFFICE_WORKER_CMD")
    if worker_cmd is not None:
        cmd = list(worker_cmd)
    elif cmd_override:
        cmd = shlex.split(cmd_override)
    else:
        cmd = [sys.executable, "-m", OFFICE_WORKER_MODULE]

    cmd.extend(["--input", str(source), "--config-json", json.dumps(worker_config)])

    if timeout_s is not None:
        cmd.extend(["--timeout-s", str(timeout_s)])

    if mem_mb is not None:
        cmd.extend(["--mem-mb", str(mem_mb)])
        
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr_tail = _tail_text(exc.stderr)
        raise TextExtractionError(
            f"office worker timed out after {timeout_s}s: reason=timeout retryable=True stderr_tail={stderr_tail}"
        ) from exc

    stderr_tail = _tail_text(result.stderr)
    stdout = (result.stdout or "").strip()
    if not stdout:
        raise TextExtractionError(
            f"office worker crash: reason=worker_crash retryable=True worker_exit_code={result.returncode} worker_stderr_tail={stderr_tail}"
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise TextExtractionError(
            f"office worker crash: reason=worker_crash retryable=True worker_exit_code={result.returncode} worker_stderr_tail={stderr_tail}"
        ) from exc

    if not isinstance(payload, dict):
        raise TextExtractionError(
            f"office worker crash: reason=worker_crash retryable=True worker_exit_code={result.returncode} worker_stderr_tail={stderr_tail}"
        )

    ok = bool(payload.get("ok", False))
    if not ok:
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        reason = str(error.get("reason", "parse_failed"))
        retryable = bool(error.get("retryable", False))
        details = error.get("details") if isinstance(error.get("details"), dict) else {}
        raise TextExtractionError(
            "office worker failed: "
            f"reason={reason} retryable={retryable} details={details} "
            f"worker_exit_code={result.returncode} worker_stderr_tail={stderr_tail}"
        )

    if result.returncode != 0:
        raise TextExtractionError(
            f"office worker crash: reason=worker_crash retryable=True worker_exit_code={result.returncode} worker_stderr_tail={stderr_tail}"
        )

    text = payload.get("text", "")
    if not isinstance(text, str):
        raise TextExtractionError(
            f"office worker crash: reason=worker_crash retryable=True worker_exit_code={result.returncode} worker_stderr_tail={stderr_tail}"
        )

    return text


class OfficeConverter:
    """Convert legacy Office documents to modern formats using headless LibreOffice."""

    def __init__(self, soffice_path: str = "soffice", default_timeout_s: int = 90):
        self.soffice_path = soffice_path
        self.default_timeout_s = default_timeout_s

    def convert(self, input_path: Path, target_ext: str, out_dir: Path, *, timeout_s: int | None = None) -> Path:
        source = validate_file(str(input_path))
        target = target_ext.lower().lstrip(".")
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s

        out_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex
        job_root = Path("/tmp/archives_scraper/office") / job_id
        lo_home = job_root / "home"
        convert_out = job_root / "out"
        lo_home.mkdir(parents=True, exist_ok=True)
        convert_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.soffice_path,
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--norestore",
            "--convert-to",
            target,
            "--outdir",
            str(convert_out),
            str(source),
        ]

        start = time.time()
        logger.info(
            "Starting LibreOffice conversion",
            extra={
                "source": str(source),
                "target_ext": target,
                "timeout_s": timeout,
            },
        )

        try:
            env = dict(os.environ)
            env["HOME"] = str(lo_home)

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=timeout,
                check=False,
                text=True,
            )

            duration_ms = int((time.time() - start) * 1000)
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                raise TextExtractionError(
                    f"libreoffice conversion failed: rc={proc.returncode}, target={target}, stderr={stderr[:400]}"
                )

            selected = self._select_output(convert_out, source, target)
            if selected.stat().st_size <= 0:
                raise TextExtractionError(f"libreoffice produced empty output: {selected}")

            output_path = out_dir / f"{source.stem}.{target}"
            shutil.copy2(selected, output_path)
            if not output_path.exists() or output_path.stat().st_size <= 0:
                raise TextExtractionError(f"failed to stage converted output: {output_path}")

            logger.info(
                "Completed LibreOffice conversion",
                extra={
                    "source": str(source),
                    "converted_to": str(output_path),
                    "target_ext": target,
                    "convert_ms": duration_ms,
                },
            )
            return output_path

        except subprocess.TimeoutExpired as exc:
            raise TextExtractionError(
                f"libreoffice conversion timed out after {timeout}s for {source}"
            ) from exc
        except FileNotFoundError as exc:
            raise TextExtractionError(
                f"LibreOffice executable not found: {self.soffice_path}"
            ) from exc
        finally:
            if _env_flag("DEBUG_KEEP_TEMPS", default=False):
                logger.warning("Preserving Office temp directory for debug: %s", job_root)
            else:
                shutil.rmtree(job_root, ignore_errors=True)

    @staticmethod
    def _select_output(convert_out: Path, source: Path, target_ext: str) -> Path:
        exact = convert_out / f"{source.stem}.{target_ext}"
        if exact.exists() and exact.is_file():
            return exact

        target_files = [
            candidate
            for candidate in convert_out.iterdir()
            if candidate.is_file() and candidate.suffix.lower().lstrip(".") == target_ext
        ]
        if not target_files:
            raise TextExtractionError(
                f"libreoffice conversion output not found for {source} in {convert_out}"
            )

        basename_matches = [f for f in target_files if f.stem.casefold() == source.stem.casefold()]
        if len(basename_matches) == 1:
            return basename_matches[0]

        if len(target_files) == 1:
            return target_files[0]

        names = ", ".join(sorted(f.name for f in target_files))
        raise TextExtractionError(
            f"multiple libreoffice outputs for {source.name}: {names}"
        )


def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def _normalize_office_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _strip_control_chars(text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


class MammothRawTextCleaner:
    """Strips the inline HTML and Markdown that mammoth.extract_raw_text() leaves behind.

    mammoth outputs mostly plain text, but leaks HTML anchor tags (e.g.
    ``<a id="_Toc...">``) and Markdown-style formatting (headings, bold, images
    with data-URI srcs, etc.).  This class handles exactly that output; it is not
    intended as a general-purpose HTML/Markdown stripper.

    For real HTML documents use ``strip_html`` instead — it uses BeautifulSoup,
    removes script/style blocks entirely, and collapses whitespace to a flat string.
    """

    # Inline HTML — any <tag> or </tag> or <tag attr="...">
    _html_tag_re = re.compile(r"<[^>]+>")
    # Markdown image with a data URI src — stripped entirely (can be millions of chars)
    _data_uri_image_re = re.compile(r"!\[[^\]]*\]\(data:[^)]*\)")
    # Regular Markdown image — stripped entirely (no useful plain-text content)
    _image_re = re.compile(r"!\[[^\]]*\]\([^)]*\)")
    # Markdown link — URL dropped, label text kept
    _link_re = re.compile(r"\[([^\]]*)\]\([^)]*\)")
    # ATX heading markers (# through ######)
    _heading_re = re.compile(r"^#{1,6}\s+", re.MULTILINE)
    # Bold (**text**) — markers removed, text kept; matched before italic
    _bold_re = re.compile(r"\*\*([^\n*]*?)\*\*")
    # Italic (*text*) — markers removed, text kept
    _italic_re = re.compile(r"\*([^\n*]*?)\*")
    # Double-underscore emphasis (__text__) — markers removed, text kept
    _double_underscore_re = re.compile(r"__([^\n_]+)__")
    # Backslash escapes e.g. \( \[ \* — backslash removed, character kept
    _backslash_escape_re = re.compile(r"\\(.)")

    def strip_html_tags(self, text: str) -> str:
        """Remove inline HTML tags, preserving all surrounding text and whitespace."""
        return self._html_tag_re.sub("", text)

    def strip_markdown(self, text: str) -> str:
        """Remove Markdown formatting marks, preserving plain text content.

        Strips data-URI images entirely (they can be millions of chars), removes
        regular images, unwraps links to their label text, removes ATX heading
        markers, unwraps bold/italic/double-underscore emphasis, and unescapes
        backslash escapes.
        """
        text = self._data_uri_image_re.sub("", text)
        text = self._image_re.sub("", text)
        text = self._link_re.sub(r"\1", text)
        text = self._heading_re.sub("", text)
        # Bold before italic so **text** doesn't partially match *text*
        text = self._bold_re.sub(r"\1", text)
        text = self._italic_re.sub(r"\1", text)
        text = self._double_underscore_re.sub(r"\1", text)
        text = self._backslash_escape_re.sub(r"\1", text)
        return text


class WordFileTextExtractor(FileTextExtractor):
    """Extract text from Word documents using mammoth/python-docx with LibreOffice fallback for .doc."""

    file_extensions: List[str] = ["docx", "docm", "doc"]
    cleaner = MammothRawTextCleaner()

    def __init__(self, converter: OfficeConverter | None = None, max_output_chars: int = 5_000_000):
        super().__init__()
        self.converter = converter or OfficeConverter()
        self.max_output_chars = max_output_chars

    def __call__(self, path: str) -> str:
        source = validate_file(path)
        ext = source.suffix.lower().lstrip(".")

        if _should_route_to_office_subprocess(source, ext):
            return run_office_worker(source, config=self._worker_config())

        method_used = ""

        if ext in ("docx", "docm"):
            text, parser_method, paragraphs_count = self._extract_docx(source)
            method_used = parser_method
        elif ext == "doc":
            with tempfile.TemporaryDirectory(prefix="office_word_") as out_dir:
                converted = self.converter.convert(
                    source,
                    "docx",
                    Path(out_dir),
                    timeout_s=90,
                )
                text, parser_method, paragraphs_count = self._extract_docx(converted)
                method_used = f"libreoffice+{parser_method}"
        else:
            raise ValueError(f"Unsupported extension for Word extractor: {ext}")

        normalized = _normalize_office_text(text)
        normalized, truncated = _truncate_text(normalized, self.max_output_chars)
        if truncated:
            logger.warning(
                "Word extraction truncated output",
                extra={"source": str(source), "max_output_chars": self.max_output_chars, "truncated": True},
            )

        logger.info(
            "Word extraction completed",
            extra={
                "source": str(source),
                "extractor": "word",
                "method_used": method_used,
                "paragraphs_count": paragraphs_count,
                "truncated": truncated,
                "text_length": len(normalized),
            },
        )
        return normalized

    def _worker_config(self) -> dict:
        return {
            "soffice_path": self.converter.soffice_path,
            "max_output_chars": self.max_output_chars,
        }

    def _extract_docx(self, path: Path) -> tuple[str, str, int]:
        mammoth_text = ""
        try:
            import mammoth

            with open(path, "rb") as file_handle:
                result = mammoth.extract_raw_text(file_handle)
            mammoth_text = (result.value or "").strip()
            if mammoth_text:
                mammoth_text = self.cleaner.strip_html_tags(mammoth_text)
                mammoth_text = self.cleaner.strip_markdown(mammoth_text)
                mammoth_text = mammoth_text.strip()
            if mammoth_text:
                paragraphs_count = sum(1 for line in mammoth_text.splitlines() if line.strip())
                return mammoth_text, "mammoth", paragraphs_count
        except Exception as exc:
            logger.warning("Mammoth extraction failed, falling back to python-docx: %s", exc)

        try:
            from docx import Document
        except Exception as exc:
            raise TextExtractionError(
                f"python-docx unavailable for fallback parsing: {path}"
            ) from exc

        doc = Document(str(path))
        paragraphs: list[str] = []
        for paragraph in doc.paragraphs:
            text = (paragraph.text or "").strip()
            if text:
                paragraphs.append(text)

        for table in doc.tables:
            for row in table.rows:
                cells = [(cell.text or "").strip() for cell in row.cells]
                if any(cells):
                    paragraphs.append("\t".join(cells))

        joined = "\n".join(paragraphs)
        return joined, "python-docx", len(paragraphs)


class PresentationTextExtractor(FileTextExtractor):
    """Extract text from modern and legacy PowerPoint files."""

    file_extensions: List[str] = ["pptx", "pptm", "ppsx", "ppt", "pps"]

    def __init__(
        self,
        include_notes: bool = True,
        max_slides: int = 500,
        max_chars_per_slide: int = 50_000,
        converter: OfficeConverter | None = None,
    ):
        super().__init__()
        self.include_notes = include_notes
        self.max_slides = max_slides
        self.max_chars_per_slide = max_chars_per_slide
        self.converter = converter or OfficeConverter(default_timeout_s=120)

    def __call__(self, path: str) -> str:
        source = validate_file(path)
        ext = source.suffix.lower().lstrip(".")

        if _should_route_to_office_subprocess(source, ext):
            return run_office_worker(source, config=self._worker_config())

        if ext in ("pptx", "pptm", "ppsx"):
            text, slide_count, shapes_scanned, truncated, notes_included = self._extract_pptx(source)
            method_used = "python-pptx"
        elif ext in ("ppt", "pps"):
            with tempfile.TemporaryDirectory(prefix="office_ppt_") as out_dir:
                converted = self.converter.convert(
                    source,
                    "pptx",
                    Path(out_dir),
                    timeout_s=120,
                )
                text, slide_count, shapes_scanned, truncated, notes_included = self._extract_pptx(converted)
                method_used = "libreoffice+python-pptx"
        else:
            raise ValueError(f"Unsupported extension for presentation extractor: {ext}")

        normalized = _normalize_office_text(text)
        logger.info(
            "Presentation extraction completed",
            extra={
                "source": str(source),
                "extractor": "presentation",
                "method_used": method_used,
                "slide_count": slide_count,
                "shapes_scanned": shapes_scanned,
                "notes_included": notes_included,
                "truncated": truncated,
                "text_length": len(normalized),
            },
        )
        return normalized

    def _worker_config(self) -> dict:
        return {
            "soffice_path": self.converter.soffice_path,
            "include_ppt_notes": self.include_notes,
            "ppt": {
                "max_slides": self.max_slides,
                "max_chars_per_slide": self.max_chars_per_slide,
            },
        }

    def _extract_pptx(self, path: Path) -> tuple[str, int, int, bool, bool]:
        try:
            from pptx import Presentation
        except Exception as exc:
            raise TextExtractionError("python-pptx is required for presentation extraction") from exc

        presentation = Presentation(str(path))
        blocks: list[str] = []
        shapes_scanned = 0
        truncated = False

        for index, slide in enumerate(presentation.slides, start=1):
            if index > self.max_slides:
                truncated = True
                logger.warning(
                    "Presentation truncated by slide limit",
                    extra={"source": str(path), "max_slides": self.max_slides, "truncated": True},
                )
                break

            slide_lines: list[str] = [f"# slide {index}"]
            text_parts: list[str] = []
            for shape in slide.shapes:
                shapes_scanned += 1
                shape_text = self._shape_text(shape)
                if shape_text:
                    text_parts.append(shape_text)

            if text_parts:
                slide_lines.extend(text_parts)

            if self.include_notes:
                notes_text = self._notes_text(slide)
                if notes_text:
                    slide_lines.append("notes:")
                    slide_lines.append(notes_text)

            slide_block = "\n".join(slide_lines).strip()
            if len(slide_block) > self.max_chars_per_slide:
                slide_block = slide_block[: self.max_chars_per_slide]
                truncated = True
                logger.warning(
                    "Presentation slide text truncated",
                    extra={
                        "source": str(path),
                        "slide_index": index,
                        "max_chars_per_slide": self.max_chars_per_slide,
                        "truncated": True,
                    },
                )

            blocks.append(slide_block)

        return "\n\n".join(blocks), min(len(presentation.slides), self.max_slides), shapes_scanned, truncated, self.include_notes

    @staticmethod
    def _shape_text(shape) -> str:
        lines: list[str] = []

        if getattr(shape, "has_text_frame", False) and shape.text_frame is not None:
            for paragraph in shape.text_frame.paragraphs:
                runs = [run.text for run in paragraph.runs if run.text]
                paragraph_text = "".join(runs).strip() if runs else (paragraph.text or "").strip()
                if paragraph_text:
                    lines.append(paragraph_text)

        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                row_cells = [(cell.text or "").strip() for cell in row.cells]
                if any(row_cells):
                    lines.append("\t".join(row_cells))

        if getattr(shape, "shape_type", None) == 6 and hasattr(shape, "shapes"):
            for grouped_shape in shape.shapes:
                nested_text = PresentationTextExtractor._shape_text(grouped_shape)
                if nested_text:
                    lines.append(nested_text)

        return "\n".join(lines).strip()

    @staticmethod
    def _notes_text(slide) -> str:
        try:
            if not getattr(slide, "has_notes_slide", False):
                return ""
            notes_frame = slide.notes_slide.notes_text_frame
            if notes_frame is None:
                return ""
            return (notes_frame.text or "").strip()
        except Exception:
            return ""


class SpreadsheetTextExtractor(FileTextExtractor):
    """Extract structured text from spreadsheets using openpyxl with LibreOffice conversion for .xls."""

    file_extensions: List[str] = ["xlsx", "xlsm", "xls"]

    def __init__(
        self,
        delimiter: str = "\t",
        max_sheets: int = 20,
        max_rows_per_sheet: int = 5_000,
        max_cols_per_sheet: int = 200,
        max_total_cells: int = 500_000,
        converter: OfficeConverter | None = None,
    ):
        super().__init__()
        self.delimiter = delimiter
        self.max_sheets = max_sheets
        self.max_rows_per_sheet = max_rows_per_sheet
        self.max_cols_per_sheet = max_cols_per_sheet
        self.max_total_cells = max_total_cells
        self.converter = converter or OfficeConverter()

    def __call__(self, path: str) -> str:
        source = validate_file(path)
        ext = source.suffix.lower().lstrip(".")

        if _should_route_to_office_subprocess(source, ext):
            return run_office_worker(source, config=self._worker_config())

        if ext in ("xlsx", "xlsm"):
            text, sheets_scanned, rows_scanned, cells_scanned, truncated = self._extract_xlsx(source)
            method_used = "openpyxl"
        elif ext == "xls":
            with tempfile.TemporaryDirectory(prefix="office_xls_") as out_dir:
                converted = self.converter.convert(
                    source,
                    "xlsx",
                    Path(out_dir),
                    timeout_s=90,
                )
                text, sheets_scanned, rows_scanned, cells_scanned, truncated = self._extract_xlsx(converted)
                method_used = "libreoffice+openpyxl"
        else:
            raise ValueError(f"Unsupported extension for spreadsheet extractor: {ext}")

        normalized = _normalize_office_text(text)
        logger.info(
            "Spreadsheet extraction completed",
            extra={
                "source": str(source),
                "extractor": "spreadsheet",
                "method_used": method_used,
                "sheets_scanned": sheets_scanned,
                "rows_scanned": rows_scanned,
                "cells_scanned": cells_scanned,
                "truncated": truncated,
                "text_length": len(normalized),
            },
        )
        return normalized

    def _worker_config(self) -> dict:
        return {
            "soffice_path": self.converter.soffice_path,
            "xlsx": {
                "max_sheets": self.max_sheets,
                "max_rows_per_sheet": self.max_rows_per_sheet,
                "max_cols_per_sheet": self.max_cols_per_sheet,
                "max_total_cells": self.max_total_cells,
            },
        }

    def _extract_xlsx(self, path: Path) -> tuple[str, int, int, int, bool]:
        try:
            from openpyxl import load_workbook
        except Exception as exc:
            raise TextExtractionError("openpyxl is required for spreadsheet extraction") from exc

        workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
        blocks: list[str] = []
        rows_scanned = 0
        cells_scanned = 0
        sheets_scanned = 0
        truncated = False

        try:
            for sheet_name in workbook.sheetnames:
                if sheets_scanned >= self.max_sheets:
                    truncated = True
                    logger.warning(
                        "Spreadsheet truncated by sheet limit",
                        extra={"source": str(path), "max_sheets": self.max_sheets, "truncated": True},
                    )
                    break

                worksheet = workbook[sheet_name]
                sheet_lines = [f"# sheet: {sheet_name}"]
                sheet_rows = 0

                for row in worksheet.iter_rows(
                    min_row=1,
                    max_row=self.max_rows_per_sheet,
                    min_col=1,
                    max_col=self.max_cols_per_sheet,
                    values_only=True,
                ):
                    row_cells: list[str] = []
                    for cell in row:
                        if cells_scanned >= self.max_total_cells:
                            truncated = True
                            break
                        row_cells.append(self._cell_to_text(cell))
                        cells_scanned += 1

                    if truncated:
                        logger.warning(
                            "Spreadsheet truncated by total cell limit",
                            extra={
                                "source": str(path),
                                "max_total_cells": self.max_total_cells,
                                "truncated": True,
                            },
                        )
                        break

                    if any(value for value in row_cells):
                        while row_cells and row_cells[-1] == "":
                            row_cells.pop()
                        sheet_lines.append(self.delimiter.join(row_cells))
                    rows_scanned += 1
                    sheet_rows += 1

                if sheet_rows >= self.max_rows_per_sheet and worksheet.max_row and worksheet.max_row > self.max_rows_per_sheet:
                    truncated = True
                    logger.warning(
                        "Spreadsheet sheet truncated by row limit",
                        extra={
                            "source": str(path),
                            "sheet": sheet_name,
                            "max_rows_per_sheet": self.max_rows_per_sheet,
                            "truncated": True,
                        },
                    )

                blocks.append("\n".join(sheet_lines))
                sheets_scanned += 1

                if truncated and cells_scanned >= self.max_total_cells:
                    break
        finally:
            workbook.close()

        return "\n\n".join(blocks), sheets_scanned, rows_scanned, cells_scanned, truncated

    @staticmethod
    def _cell_to_text(value) -> str:
        if value is None:
            return ""
        return str(value).strip()
