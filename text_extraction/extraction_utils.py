# text_extraction/extraction_utils.py

# --- imports ---
import logging
import pythoncom
import subprocess
import tempfile
import unicodedata
import win32com.client
from bs4 import BeautifulSoup
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from unidecode import unidecode  # nicer fallback for weird glyphs
    _HAS_UNIDECODE = True
except ImportError:
    _HAS_UNIDECODE = False

# common replacements (curly quotes, dashes, ligatures, etc.)
def common_char_replacements(text: str) -> str:
    """
    Replace common typographic Unicode characters with simpler ASCII equivalents.

    Parameters
    ----------
    text : str
        Input string possibly containing curly quotes, dashes, ligatures, etc.

    Returns
    -------
    str
        Text with characters like “ ” – — ﬁ ﬂ replaced by their ASCII counterparts.
    """

    replacements_dict = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u00a0": " ",  # non-breaking space
        "\u2026": "...",  # ellipsis
        "\ufb01": "fi",  # ﬁ ligature
        "\ufb02": "fl",  # ﬂ ligature
        "\x00": "",  # remove NUL bytes
    }
    for src, dst in replacements_dict.items():
        text = text.replace(src, dst)   
    return text

def strip_diacritics(text: str) -> str:
    """
    Remove diacritical marks from the input text, optionally transliterating
    exotic glyphs to ASCII.

    Parameters
    ----------
    text : str
        Input string possibly containing accented characters.

    Returns
    -------
    str
        Text with diacritics stripped; uses unidecode if available, else drops
        non-ASCII characters.
    """
    # Normalize to NFD to separate base chars from diacritics
    nfkd = unicodedata.normalize("NFD", text)
    # Remove combining marks (diacritics)
    no_diacritics = "".join(c for c in nfkd if not unicodedata.category(c).startswith("M"))
    # Recompose
    cleaned = unicodedata.normalize("NFC", no_diacritics)
    if _HAS_UNIDECODE:
        # Further transliterate any remaining exotic characters to ASCII
        cleaned = unidecode(cleaned)
    else:
        # Drop any remaining non-ASCII aggressively
        cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii")
    return cleaned

def validate_file(path: str) -> Path:
    """
    Ensure the given path exists and is a file.

    Parameters
    ----------
    path : str
        Filesystem path to validate.

    Returns
    -------
    Path
        A pathlib.Path object for the valid file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a file.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(path)
    return p

def normalize_whitespace(text: str) -> str:
    """
    Collapse all whitespace (spaces, newlines, tabs) into single spaces.

    Parameters
    ----------
    text : str
        Input text to normalize.

    Returns
    -------
    str
        Text with all runs of whitespace replaced by a single space.
    """
    return " ".join(text.split())

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text to NFC form.

    Parameters
    ----------
    text : str
        Input text to normalize.

    Returns
    -------
    str
        NFC-normalized text.
    """
    return unicodedata.normalize("NFC", text)

def strip_html(html: str, parser: str = "lxml", remove_tags=None) -> str:
    """
    Strip HTML tags and collapse resulting text.

    Parameters
    ----------
    html : str
        Raw HTML content.
    parser : str, optional
        Parser to pass to BeautifulSoup, by default "lxml".
    remove_tags : list[str] or None
        Tags to remove entirely (e.g., ["script", "style"]), by default None.

    Returns
    -------
    str
        Clean text with HTML removed and whitespace normalized.
    """
    if remove_tags is None:
        remove_tags = ["script", "style", "noscript"]
    soup = BeautifulSoup(html, parser)
    for t in soup(remove_tags):
        t.decompose()
    return normalize_whitespace(soup.get_text(separator=" ", strip=True))

def run_pandoc(src: str, pandoc_path: str, to_format: str = "plain") -> Path:
    """
    Convert a document using Pandoc and return the path to the output file.

    Parameters
    ----------
    src : str
        Source document path.
    pandoc_path : str
        Full path to the pandoc executable.
    to_format : str, optional
        Output format for pandoc (default is "plain").

    Returns
    -------
    Path
        Path to the converted output file.

    Raises
    ------
    subprocess.CalledProcessError
        If pandoc fails.
    """
    out = Path(tempfile.mkdtemp()) / (Path(src).stem + ".txt")
    cmd = [pandoc_path, src, "-t", to_format, "-o", str(out)]
    subprocess.run(cmd, check=True)
    return out

@contextmanager
def com_app(dispatch_name: str, visible: bool = False):
    """
    Context manager for controlling a COM application (e.g., Word, PowerPoint).

    Parameters
    ----------
    dispatch_name : str
        ProgID for the COM application (e.g., "Word.Application").
    visible : bool, optional
        Whether the COM window is visible, by default False.

    Yields
    ------
    COM object
        The initialized COM application instance.

    Notes
    -----
    Ensures CoInitialize/CoUninitialize around the COM session.
    """
    pythoncom.CoInitialize()
    app = win32com.client.DispatchEx(dispatch_name)
    app.Visible = visible
    try:
        yield app
    finally:
        app.Quit()
        pythoncom.CoUninitialize()