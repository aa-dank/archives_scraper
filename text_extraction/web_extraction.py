# text_extraction/web_extractor.py

import email
import logging
from pathlib import Path
from typing import List
from email import policy

from .basic_extraction import FileTextExtractor
from .extraction_utils import validate_file, strip_html

logger = logging.getLogger(__name__)

class HtmlTextExtractor(FileTextExtractor):
    """
    Extract text content from HTML and MHTML files.

    This class implements text extraction from HTML-based file formats, including
    `.html`, `.htm`, `.mhtml`, and `.mht`. It uses BeautifulSoup for parsing and
    cleaning the HTML content, removing unnecessary tags like `<script>`, `<style>`,

    Attributes
    ----------
    file_extensions : List[str]
        Supported file extensions for HTML-based files.
    parser : str
        The parser to use with BeautifulSoup, e.g., "lxml" or "html.parser".

    Methods
    -------
    __call__(path: str) -> str
        Extract text content from the given HTML or MHTML file.
    _extract_from_mhtml(path: Path) -> str
        Extract HTML content from an MHTML file.
    """
    # Extensions are lowercase, no leading dot (as per spec)
    file_extensions: List[str] = ["html", "htm", "mhtml", "mht"]

    def __init__(self, parser: str = "lxml"):
        """
        Initialize the HtmlTextExtractor.

        Parameters
        ----------
        parser : str, optional
            The parser to use with BeautifulSoup, by default "lxml".
        """
        super().__init__()
        self.parser = parser

    def __call__(self, path: str) -> str:
        """
        Extract text content from an HTML or MHTML file.

        Parameters
        ----------
        path : str
            Path to the HTML or MHTML file.

        Returns
        -------
        str
            Extracted text content from the file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist or is not a valid file.
        """
        logger.info(f"Extracting text from HTML file: {path}")
        # validate file existence and type
        p = validate_file(path)
        logger.debug(f"Validated HTML file path: {p}")

        ext = p.suffix.lower().lstrip(".")
        logger.debug(f"HTML file extension detected: {ext}")
        if ext in ("mhtml", "mht"):
            html = self._extract_from_mhtml(p)
        else:
            html = p.read_text(errors="ignore", encoding="utf-8")

        # use shared HTML stripping utility
        return strip_html(html, parser=self.parser)

    def _extract_from_mhtml(self, path: Path) -> str:
        """
        Extract HTML content from an MHTML file.

        Parameters
        ----------
        path : Path
            Path to the MHTML file.

        Returns
        -------
        str
            Extracted HTML content from the MHTML file.

        Notes
        -----
        If no `text/html` part is found in the MHTML file, an empty string is returned.
        """
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        # Find the first text/html part
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                return part.get_content()
        return ""
    

class EmailTextExtractor(FileTextExtractor):
    """
    Extract text content from email files.

    This class implements text extraction from email files, handling both plain text
    and HTML content. It uses the `email` module to parse the email structure.

    Attributes
    ----------
    file_extensions : List[str]
        Supported file extensions for email files.

    Methods
    -------
    __call__(path: str) -> str
        Extract text content from the given email file.
    """
    # Extensions are lowercase, no leading dot (as per spec)
    file_extensions: List[str] = ["eml", "msg"]

    def __init__(self, parser: str = "lxml"):
        """
        Initialize the EmailTextExtractor.

        Parameters
        ----------
        parser : str, optional
            The parser to use with BeautifulSoup for HTML content, by default "lxml".
        """
        super().__init__()
        self.parser = parser
        from .extraction_utils import normalize_whitespace

    def __call__(self, path: str) -> str:
        """
        Extract text content from an email file.

        Parameters
        ----------
        path : str
            Path to the email file.

        Returns
        -------
        str
            Extracted text content from the email file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist or is not a valid file.
        """
        logger.info(f"Extracting text from email file: {path}")
        # validate file
        from .extraction_utils import validate_file, strip_html, normalize_whitespace
        p = validate_file(path)
        logger.debug(f"Validated email file path: {p}")

        with open(p, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        text_parts = []
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                text_parts.append(part.get_content())
            elif content_type == "text/html":
                html = part.get_content()
                # use shared HTML stripping
                text_parts.append(strip_html(html, parser=self.parser))

        # normalize overall whitespace
        combined = " ".join(text_parts)
        return normalize_whitespace(combined)
