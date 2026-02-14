# text_extraction/image_extraction.py
import logging
import json
import os
import pytesseract
import re
import subprocess
import sys
import tempfile
from typing import List, Tuple
from pathlib import Path
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

IMAGE_OCR_PIXEL_THRESHOLD = 30_000_000
IMAGE_OCR_FILESIZE_THRESHOLD_BYTES = 50 * 1024 * 1024
IMAGE_OCR_SUBPROCESS_TIMEOUT_S = int(os.getenv("IMAGE_OCR_SUBPROCESS_TIMEOUT_S", "300"))
_IMAGE_OCR_SUBPROCESS_MEM_MB_RAW = os.getenv("IMAGE_OCR_SUBPROCESS_MEM_MB")
IMAGE_OCR_SUBPROCESS_MEM_MB = int(_IMAGE_OCR_SUBPROCESS_MEM_MB_RAW) if _IMAGE_OCR_SUBPROCESS_MEM_MB_RAW else None
IMAGE_OCR_SUBPROCESS_STDERR_MAX_BYTES = 8192

from .basic_extraction import FileTextExtractor
from .extraction_utils import ImageOCRUtils, OCRUtils


class ImageTextExtractor(FileTextExtractor):
    """
    OCR text from image files using Tesseract (via pytesseract).

    Supports automatic orientation correction via Tesseract OSD,
    plus optional light pre-processing for better OCR on scans/phone pics.
    
    Supports: PNG, JPG/JPEG, TIFF, BMP, GIF (first frame), HEIC (if pillow-heif installed).
    """
    # Extensions are lowercase, no leading dot (as per spec)
    file_extensions: List[str] = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"]

    def __init__(self,
                 lang: str = "eng",
                 tesseract_cmd: str | None = None,
                 psm: int = 3,
                 oem: int = 3,
                 preprocess: bool = True,
                 max_side: int = 3000,
                 default_image_dpi: int = 300):
        r"""
        Parameters
        ----------
        lang : str
            Tesseract language(s). e.g. "eng+spa".
        tesseract_cmd : str | None
            Full path to tesseract.exe if not on PATH. (eg r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        psm : int
            Page segmentation mode. 3 = fully automatic, 6 = assume uniform blocks of text.
        oem : int
            OCR Engine mode. 3 = default, based on what is available.
        preprocess : bool
            Whether to apply grayscale/threshold/denoise pre-processing.
        max_side : int
            Resize largest image side to this (keeps memory reasonable).
        default_image_dpi : int
            DPI to use for images without embedded DPI info.
        """
        super().__init__()
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.tesseract_cmd = tesseract_cmd
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.preprocess = preprocess
        self.max_side = max_side
        self.default_image_dpi = default_image_dpi

    def __call__(self, path: str) -> str:
        logger.info(f"Extracting text from image: {path}")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        file_size = p.stat().st_size
        width, height, frame_count = self._inspect_image(path=p)
        pixel_count = width * height

        should_use_subprocess = (
            frame_count > 1
            or pixel_count > IMAGE_OCR_PIXEL_THRESHOLD
            or file_size > IMAGE_OCR_FILESIZE_THRESHOLD_BYTES
        )

        if should_use_subprocess:
            logger.info(
                "Using image OCR subprocess: path=%s frame_count=%s pixels=%s file_size=%s",
                p,
                frame_count,
                pixel_count,
                file_size,
            )
            return self._run_ocr_subprocess(path=p, worker_config=self._worker_config())

        return self._extract_in_process(path=p)

    def _extract_in_process(self, path: Path | str) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(path))

        images = self._load_images(p)
        logger.debug(f"Loaded {len(images)} image frames for OCR")
        texts = []
        for img in images:
            # detect and correct orientation
            img = self._ensure_longside_bottom(img)
            img = self._inject_dpi(img, self.default_image_dpi)
            img = self.detect_and_correct_orientation(img)
            if self.preprocess:
                img = self._preprocess(img)
                logger.debug("Applied preprocessing to image")

            txt = pytesseract.image_to_string(
                image=img,
                lang=self.lang,
                config=config_str(f"--psm {self.psm}", f"--oem {self.oem}")
                )
            logger.debug(f"Extracted text length: {len(txt)} characters")
            texts.append(txt)

        return "\n".join(texts)

    @staticmethod
    def _inspect_image(path: Path) -> Tuple[int, int, int]:
        return ImageOCRUtils.inspect_image(path, stop_after_frames=2)

    @staticmethod
    def _read_stderr_snippet(stderr_file, max_bytes: int = IMAGE_OCR_SUBPROCESS_STDERR_MAX_BYTES) -> str:
        stderr_file.flush()
        stderr_file.seek(0)
        stderr_bytes = stderr_file.read(max_bytes + 1)
        if not stderr_bytes:
            return ""

        is_truncated = len(stderr_bytes) > max_bytes
        snippet = stderr_bytes[:max_bytes].decode("utf-8", errors="replace").strip()
        if is_truncated:
            return f"{snippet}...[truncated]"
        return snippet

    @staticmethod
    def _run_ocr_subprocess(
        path: Path,
        worker_config: dict,
        timeout_s: int = IMAGE_OCR_SUBPROCESS_TIMEOUT_S,
        mem_mb: int | None = IMAGE_OCR_SUBPROCESS_MEM_MB,
    ) -> str:
        cmd = [
            sys.executable,
            "-m",
            "text_extraction.image_extraction_worker",
            "--input",
            str(path),
            "--config-json",
            json.dumps(worker_config),
        ]

        if mem_mb is not None:
            cmd.extend(["--mem-mb", str(mem_mb)])

        with tempfile.NamedTemporaryFile(prefix="image_ocr_subprocess_stderr_", suffix=".log") as stderr_file:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                stderr_snippet = ImageTextExtractor._read_stderr_snippet(stderr_file)
                logger.error(
                    "Image OCR subprocess timed out: path=%s timeout_s=%s stderr=%s",
                    path,
                    timeout_s,
                    stderr_snippet,
                )
                raise RuntimeError("image ocr subprocess timed out")

            if result.returncode == 0:
                return result.stdout

            stderr_snippet = ImageTextExtractor._read_stderr_snippet(stderr_file)
            logger.error(
                "Image OCR subprocess failed: path=%s returncode=%s stderr=%s",
                path,
                result.returncode,
                stderr_snippet,
            )

            if result.returncode == -9:
                raise RuntimeError("image ocr subprocess killed (likely OOM)")

            raise RuntimeError(
                f"image ocr subprocess failed (returncode={result.returncode}): {stderr_snippet}"
            )

    def _worker_config(self) -> dict:
        return {
            "lang": self.lang,
            "psm": self.psm,
            "oem": self.oem,
            "preprocess": self.preprocess,
            "max_side": self.max_side,
            "default_image_dpi": self.default_image_dpi,
            "tesseract_cmd": self.tesseract_cmd,
        }

    # ---------- helpers ----------
    def _load_images(self, path: Path) -> List[Image.Image]:
        """Handle multi-page TIFFs and GIFs gracefully."""
        logger.debug(f"Loading images from path: {path}")
        return ImageOCRUtils.load_images(path=path, max_side=self.max_side)

    def _preprocess(self, pil_img: Image.Image) -> Image.Image:
        """
        Simple preprocessing:
          - convert to grayscale
          - optional OpenCV adaptive threshold / denoise if available
        """
        try:
            import cv2  # optional for better preprocessing
            import numpy as np
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        logger.debug("Starting preprocessing of image, _HAS_CV2=%s", has_cv2)
        if has_cv2:
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
            # adaptive threshold helps on uneven lighting
            img = cv2.adaptiveThreshold(img, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 10)
            return Image.fromarray(img)
        else:
            # Pillow-only fallback
            img = ImageOps.grayscale(pil_img)
            # Simple point threshold
            img = img.point(lambda x: 255 if x > 200 else 0)
            return img

    def _ensure_longside_bottom(self, pil_img: Image.Image) -> Image.Image:
        """
        Rotate image to landscape if its short/long ratio deviates from
        8.5×11 (≈0.773), so the long side ends up on the bottom.
        """
        rotated = ImageOCRUtils.ensure_longside_bottom(pil_img)
        if rotated is not pil_img:
            w, h = pil_img.size
            logger.debug(f"Rotating image from portrait to landscape: {w}x{h}")
        return rotated

    def detect_and_correct_orientation(self, pil_img: Image.Image) -> Image.Image:
        """
        Use Tesseract OSD to detect rotation and counter-rotate image upright.
        """
        try:
            osd = pytesseract.image_to_osd(pil_img)
        except pytesseract.TesseractError as e:
            logger.error(f"Tesseract OSD failed: {e}")
            return pil_img

        logger.debug(f"Tesseract OSD output: {osd.strip()}")
        rot_match = re.search(r"Rotate: (\d+)", osd)
        if rot_match:
            angle = int(rot_match.group(1))
            if angle != 0:
                pil_img = pil_img.rotate(360 - angle, expand=True)
                logger.info(f"Rotated image by {360-angle} degrees to correct orientation")
        return pil_img
    
    def _inject_dpi(self, pil_img: Image.Image, dpi: int) -> Image.Image:
        """
        Inject DPI into the image metadata if not present.
        """
        existing = pil_img.info.get("dpi", (0, 0))[0]
        if not existing:
            logger.debug(f"Injecting default DPI {dpi} into image")
        return ImageOCRUtils.inject_dpi(pil_img, dpi)


# Small utility so we can extend config easily
def config_str(*parts: str) -> str:
    return OCRUtils.config_str(*parts)
