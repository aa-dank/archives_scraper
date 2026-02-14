# text_extraction/extraction_utils.py

# --- imports ---
import logging
import re
import sys
import warnings
from contextlib import contextmanager
from typing import Optional, Tuple
import subprocess
import tempfile
import unicodedata
from bs4 import BeautifulSoup
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

def init_tesseract(cmd: Optional[str] = None):
    """Configure pytesseract to use a specific Tesseract executable if provided.

    Parameters:
        cmd (Optional[str]): Full path to tesseract binary; uses PATH if None.
    """
    import pytesseract
    # Set custom tesseract command if given
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


@contextmanager
def pil_decompression_bomb_as_error():
    """Treat `PIL.Image.DecompressionBombWarning` as an exception within the context."""
    try:
        from PIL import Image

        bomb_warning = Image.DecompressionBombWarning
    except Exception:
        bomb_warning = None

    with warnings.catch_warnings():
        if bomb_warning is not None:
            warnings.simplefilter("error", bomb_warning)
        yield


def is_pil_decompression_bomb(exc: BaseException) -> bool:
    """Return True if the exception is (or looks like) a Pillow decompression-bomb."""
    try:
        from PIL import Image

        return isinstance(exc, (Image.DecompressionBombError, Image.DecompressionBombWarning))
    except Exception:
        # Fall back to string matching if Pillow isn't importable for some reason.
        name = exc.__class__.__name__
        msg = str(exc)
        return "DecompressionBomb" in name or "decompression bomb" in msg.lower()


class SubprocessUtils:
    """Helpers for subprocess workers and process-level limits."""

    @staticmethod
    def apply_memory_limit(mem_mb: Optional[int], stderr=None) -> None:
        """Apply a Linux address-space memory limit (best-effort)."""
        if mem_mb is None:
            return

        err_stream = stderr if stderr is not None else sys.stderr

        if not sys.platform.startswith("linux"):
            print("warning: --mem-mb is only supported on Linux; ignoring", file=err_stream)
            return

        try:
            import resource

            limit_bytes = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception as exc:
            print(f"warning: failed to set memory limit: {exc}", file=err_stream)


class OCRUtils:
    """Shared OCR string/config helpers."""

    @staticmethod
    def config_str(*parts: str) -> str:
        return " ".join(part for part in parts if part)


class ImageOCRUtils:
    """Reusable image OCR helpers shared by image extractor and worker."""

    @staticmethod
    def inspect_image(path: Path, stop_after_frames: int = 2) -> Tuple[int, int, int]:
        from PIL import Image, ImageSequence

        with Image.open(path) as im:
            width, height = im.size
            frame_count = 0
            try:
                for _ in ImageSequence.Iterator(im):
                    frame_count += 1
                    if frame_count >= stop_after_frames:
                        break
            except Exception:
                frame_count = 1

        if frame_count == 0:
            frame_count = 1
        return width, height, frame_count

    @staticmethod
    def load_images(path: Path, max_side: int):
        from PIL import Image, ImageSequence

        images = []
        with Image.open(path) as im:
            try:
                for frame in ImageSequence.Iterator(im):
                    images.append(frame.convert("RGB"))
            except Exception:
                images.append(im.convert("RGB"))

        out = []
        for image in images:
            if max(image.size) > max_side:
                scale = max_side / max(image.size)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.LANCZOS)
            out.append(image)
        return out

    @staticmethod
    def preprocess_light(pil_img):
        from PIL import ImageOps

        image = ImageOps.grayscale(pil_img)
        return image.point(lambda pixel: 255 if pixel > 200 else 0)

    @staticmethod
    def ensure_longside_bottom(pil_img):
        width, height = pil_img.size
        short_side, long_side = sorted((width, height))
        ratio = short_side / long_side
        letter_ratio = 8.5 / 11
        if abs(ratio - letter_ratio) > 0.05 and height > width:
            return pil_img.rotate(90, expand=True)
        return pil_img

    @staticmethod
    def detect_and_correct_orientation(pil_img):
        import pytesseract

        try:
            osd = pytesseract.image_to_osd(pil_img)
        except pytesseract.TesseractError:
            return pil_img

        rotation_match = re.search(r"Rotate: (\d+)", osd)
        if rotation_match:
            angle = int(rotation_match.group(1))
            if angle != 0:
                pil_img = pil_img.rotate(360 - angle, expand=True)
        return pil_img

    @staticmethod
    def inject_dpi(pil_img, dpi: int):
        existing = pil_img.info.get("dpi", (0, 0))[0]
        if not existing:
            pil_img.info["dpi"] = (dpi, dpi)
        return pil_img

