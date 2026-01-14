# text_extraction/extractors.py

import httpx
import logging
import os
import markdown
import re
from abc import ABC, abstractmethod
from datetime import datetime, date
from pathlib import Path
from .extraction_utils import validate_file, strip_html
from typing import List

logger = logging.getLogger(__name__)

class FileTextExtractor(ABC):
    """
    Abstract base class for text extraction from different file types.
    
    This class defines the interface for all text extractors. Subclasses should
    implement the __call__ method to handle specific file formats.
    """
    file_extensions: List[str] = None  # Class variable to define supported file extensions

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.file_extensions is None:
            raise TypeError(f"Class {cls.__name__} must define 'file_extensions' class variable")
        
    @abstractmethod
    def __call__(self, path: str) -> str:
        """
        Extract text content from a file.
        
        Parameters
        ----------
        path : str
            Path to the file from which to extract text.
            
        Returns
        -------
        str
            Extracted text content from the file.
        
        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

  
class TextFileTextExtractor(FileTextExtractor):
    """
    Extract text from plain text files.
    
    This class implements text extraction from various text-based file formats
    like .txt, .md, .csv, etc. It handles different encodings and provides
    basic error handling.
    """
    file_extensions = ['txt', 'md', 'log', 'csv', 'json', 'xml', 'yaml', 'yml', 'ini', 'cfg', 'conf']
    
    def __init__(self):
        super().__init__()
        self.encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
    
    def __call__(self, path: str) -> str:
        """
        Extract text content from a plain text file.
        
        Parameters
        ----------
        path : str
            Path to the text file from which to extract text.
            
        Returns
        -------
        str
            Extracted text content from the file.
            
        Raises
        ----
        FileNotFoundError
            If the text file does not exist.
        ValueError
            If the file cannot be read with any of the supported encodings.
        """
        logger.info(f"Extracting text from file: {path}")
        # validate file path and type
        file_path = validate_file(path)
        logger.debug(f"Validated file path: {file_path}")
        
        # Try different encodings
        for encoding in self.encodings:
            logger.debug(f"Trying encoding: {encoding} for file: {file_path}")
            try:
                with open(file_path, 'r', encoding=encoding) as file: #TODO:  errors='ignore'?
                    if file_path.suffix.lower() == ".xml":
                        logger.debug(f"Stripping XML content from file: {file_path}")
                        return strip_html(file.read(), parser="xml")
                    
                    elif file_path.suffix.lower() == ".md":
                        logger.debug(f"Converting Markdown to HTML for file: {file_path}")
                        text = markdown.markdown(file.read())
                        return strip_html(text, parser="html")

                    return file.read()
            except UnicodeDecodeError:
                continue
        
        # If we get here, none of the encodings worked
        raise ValueError(f"Unable to read file with supported encodings: {path}")


class TikaUnsupportedError(Exception):
    """Raised when Tika cannot process a file due to unsupported format or encryption."""
    def __init__(self, filepath: str, message: str = "Unsupported by Tika"):
        self.filepath = filepath
        super().__init__(f"{message}: {filepath}")


class TikaNoContentError(Exception):
    """Raised when Tika returns 204 No Content (e.g., image‐only file without OCR)."""
    def __init__(self, filepath: str, message: str = "No content found"):
        self.filepath = filepath
        super().__init__(f"{message}: {filepath}")


class TikaTextExtractor(FileTextExtractor):
    """
    Fallback extractor using a containerized Apache Tika (REST API).
    """
    # catch‐all for most formats; register this last in your extractor list
    file_extensions = [
        'pdf','doc','docx','ppt','pptx','xls','xlsx','rtf',
        'html','htm','txt','csv','xml','json','md',
        'png','jpg','jpeg','gif','tif','tiff','eml','msg',
        'odt','ods'
    ]

    def __init__(self, server_url: str | None = None, timeout: int = 60):
        super().__init__()
        # e.g. "http://localhost:9998"
        self.server_url = server_url or os.environ.get('TIKA_SERVER_URL', 'http://localhost:9998')
        self.tika_endpoint = f"{self.server_url}/tika"
        self.detect_endpoint = f"{self.server_url}/detect/stream"
        self.timeout = timeout

        # sanity check server is up
        r = httpx.get(self.tika_endpoint, headers={'Accept': 'text/plain'}, timeout=self.timeout)
        r.raise_for_status()

    def _detect_mime(self, path: Path) -> str:
        # filename hint improves detection
        with open(path, 'rb') as fh:
            r = httpx.put(
                self.detect_endpoint,
                content=fh,
                headers={'Content-Disposition': f'attachment; filename=\"{path.name}\"'},
                timeout=self.timeout
            )
        r.raise_for_status()
        return (r.text or '').strip()

    def __call__(self, path: str) -> str:
        p = validate_file(path)
        # Preflight: detect MIME
        mime = self._detect_mime(p)
        logger.debug(f"Tika detected MIME for {p}: {mime or 'UNKNOWN'}")

        # Fast-fail on clearly unknown/opaque types
        if not mime or mime == 'application/octet-stream':
            raise TikaUnsupportedError(f"Tika can’t determine a usable MIME type for {p}")

        logger.info(f"Extracting text from {p} with Tika (MIME={mime})")
        # Extract text
        with open(p, 'rb') as fh:
            resp = httpx.put(
                self.tika_endpoint,
                content=fh,
                headers={'Accept': 'text/plain'},
                timeout=self.timeout
            )

        # Explicit handling of common outcomes
        if resp.status_code == 204:
            raise TikaNoContentError(f"Tika returned 204 No Content for {p}")
        if resp.status_code == 422:
            raise TikaUnsupportedError(f"Tika returned 422 (unsupported/encrypted) for {p}: {resp.text}")

        resp.raise_for_status()

        text = resp.text or ""
        if not text.strip():
            logger.warning(f"Tika returned 200 but empty body for {p} (MIME={mime})")
        return text
    
def get_extractor_for_file(file_path: str, extractors: list) -> FileTextExtractor:
    """
    Determine the appropriate extractor for a given file based on its extension.

    Parameters
    ----------
    file_path : str
        Path to the file to be processed.
    extractors : list
        List of extractor instances.

    Returns
    -------
    FileTextExtractor
        The extractor instance that matches the file extension.

    Raises
    ------
    ValueError
        If no extractor matches the file extension.
    """
    logger.debug(f"Finding extractor for file: {file_path}")
    file_extension = Path(file_path).suffix.lower().lstrip(".")
    for extractor in extractors:
        if file_extension in extractor.file_extensions:
            logger.debug(f"Selected extractor {extractor.__class__.__name__} for file: {file_path}")
            return extractor
    logger.error(f"No extractor found for file extension: {file_extension}")
    return None


class DateExtractor:
    """
    Extract explicit, absolute dates from OCR'ed construction docs.
    Supported formats (4-digit or 2-digit years):
      - YYYY[-/.]MM[-/.]DD           e.g., 2024-06-05, 2019/12/31
      - MM[-/.]DD[-/.]YYYY           e.g., 6/5/2024, 06-05-2024
      - MM[-/.]DD[-/.]YY             e.g., 6/1/24, 01/05/00  (2-digit year w/ pivot)
      - MonthName DD[, ]YYYY         e.g., Jan 5 2024, January 5, 2023
      - (Optional) DD[-/.]MM[-/.]YYYY (DMY) if you truly have it

    Year handling:
      - Two-digit years normalized via a pivot (default 60): 00–59→2000–2059; 60–99→1960–1999.
      - Final year window filter (default 1960–2035) reduces OCR noise further.
    """

    def __init__(self, year_min=1960, year_max=2035, enable_dmy=False, yy_pivot=60):
        self.year_min = year_min
        self.year_max = year_max
        self.enable_dmy = enable_dmy
        self.yy_pivot = yy_pivot  # e.g., 60 -> 60–99 => 1900s; 00–59 => 2000s

        # YYYY[-/.]MM[-/.]DD  (ISO-ish)
        self.rx_iso = re.compile(
            r'\b((?:19|20)\d{2})[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])\b'
        )

        # MM[-/.]DD[-/.]YYYY (US MDY, 4-digit year)
        self.rx_mdy4 = re.compile(
            r'\b(0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])[-/.]((?:19|20)\d{2})\b'
        )

        # NEW: MM[-/.]DD[-/.]YY (US MDY, 2-digit year)
        # Examples (match): "6/1/24", "06-01-00", "12.31.69"
        # Non-matches: "1/8" (no 2-digit year), "13/01/24" (invalid month; date() filter will reject anyway)
        self.rx_mdy2 = re.compile(
            r'\b(0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])[-/.](\d{2})\b'
        )

        # (Optional) DD[-/.]MM[-/.]YYYY (DMY)
        self.rx_dmy4 = re.compile(
            r'\b(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.]((?:19|20)\d{2})\b'
        )

        # MonthName DD[, ]YYYY
        self.rx_mon = re.compile(
            r'(?ix)\b'
            r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
            r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|'
            r'nov(?:ember)?|dec(?:ember)?)'
            r'\s+([0-3]?\d)(?:,)?\s+((?:19|20)\d{2})\b'
        )

        self.month_to_number_map = {
            m: i for i, m in enumerate(
                ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], start=1
            )
        }

    def _normalize_yy(self, yy_str: str) -> int:
        """
        Normalize a 2-digit year to 4 digits using a pivot.
        - If yy >= pivot -> 1900 + yy  (e.g., 69->1969 when pivot=60)
        - Else             2000 + yy  (e.g., 00->2000, 24->2024)
        """
        yy = int(yy_str)
        if yy >= self.yy_pivot:
            return 1900 + yy
        return 2000 + yy

    @staticmethod
    def _safe_date(y: int, m: int, d: int):
        try:
            return date(y, m, d)
        except ValueError:
            return None

    def __call__(self, txt: str):
        if not txt:
            return []

        candidates = []

        # ISO YYYY-MM-DD
        for y, m, d in self.rx_iso.findall(txt):
            candidates.append((int(y), int(m), int(d)))

        # MDY with 4-digit year
        for m, d, y in self.rx_mdy4.findall(txt):
            candidates.append((int(y), int(m), int(d)))

        # MDY with 2-digit year (normalize via pivot)
        for m, d, yy in self.rx_mdy2.findall(txt):
            y_full = self._normalize_yy(yy)  # ensures 00 -> 2000, 69 -> 1969, etc.
            candidates.append((int(y_full), int(m), int(d)))

        # DMY (optional)
        if self.enable_dmy:
            for d, m, y in self.rx_dmy4.findall(txt):
                candidates.append((int(y), int(m), int(d)))

        # MonthName DD, YYYY
        for mon, d, y in self.rx_mon.findall(txt):
            candidates.append((int(y), self.month_to_number_map[mon[:3].lower()], int(d)))

        # Validate + year window filter
        out = []
        for y, m, d in candidates:
            dt = self._safe_date(y, m, d)
            if dt and self.year_min <= dt.year <= self.year_max:
                out.append(dt)

        return out
