# text_extraction/pdf_extractor.py

import fitz
import logging
import ocrmypdf
import os
import shutil
import tempfile

from pathlib import Path
from typing import Union, List
from .basic_extraction import FileTextExtractor, TextExtractionError
from .extraction_utils import (
    validate_file,
    pil_decompression_bomb_as_error,
    is_pil_decompression_bomb,
)


class DecompressionBombTextExtractionError(TextExtractionError):
    """Raised when Pillow's decompression bomb protection triggers."""

logger = logging.getLogger(__name__)

OCR_RASTER_MEM_BUDGET_MB = int(os.getenv("OCR_RASTER_MEM_BUDGET_MB", "768"))
OCR_RASTER_OVERHEAD = 3.0
OCR_DPI_MIN = 72
OCR_DPI_MAX = 300
OCR_MAX_IMAGE_MPIXELS_LARGE_FORMAT = 80
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "25"))

class PDFFile:
    """
    Represents a PDF file and provides properties and utilities
    to inspect its content and layout.

    Attributes
    ----------
    path : Path
        Filesystem path to the PDF.
    name : str
        File name without its extension.
    page_count : int
        Total number of pages in the document.
    is_encrypted : bool
        True if the PDF is encrypted.
    property_cache : dict
        Cache for storing computed properties (e.g., page dimensions).

    Methods
    -------
    pt_to_in(pt: float) -> float
        Convert a measurement from PDF points to inches.
    _is_large_format_page(w: float, h: float, long_edge_thresh: int = 24,
                          area_thresh: int = 800) -> bool
        Determine if a page size in inches exceeds large‐format thresholds.

    Properties
    ----------
    pages_dims : List[Tuple[float, float]]
        List of (width, height) of each page in inches.
    has_large_format : bool
        True if any page qualifies as a large‐format page.
    """
    
    def __init__(self, path: str):
        """
        Initialize a PDFFile instance.

        Parameters
        ----------
        path : str
            Path to the PDF file.

        Raises
        ------
        FileNotFoundError
            If the path does not exist or is not a file.
        ValueError
            If the file cannot be opened as a PDF.
        """
        logger.debug(f"Initializing PDFFile for path: {path}")
        self.path = Path(path)
        # if the path doesn't exist or is not a file, raise an error
        if not self.path.exists(): 
            logger.error(f"PDF file not found: {self.path}")
            raise FileNotFoundError(f"PDF file not found: {path}")

        if not self.path.is_file():
            logger.error(f"PDF path is not a file: {self.path}")
            raise FileNotFoundError(f"PDF file is not a file: {path}")

        with fitz.open(self.path) as doc:
            # if the file is not a PDF, raise an error
            if not doc.is_pdf:
                logger.error(f"File is not a valid PDF: {self.path}")
                raise ValueError(f"File is not a valid PDF: {self.path}")
            
            self.page_count = doc.page_count
            self.is_encrypted = doc.is_encrypted
        logger.debug(f"PDFFile {self.path} has {self.page_count} pages; encrypted={self.is_encrypted}")

        self.name = self.path.stem
        self.size = self.path.stat().st_size  # size in bytes
        # cache for properties that are expensive to compute and not used much
        self.property_cache = {}

    @staticmethod
    def _is_large_format_page(w, h, long_edge_thresh=24, area_thresh=800):
        """
        Check if a page size exceeds defined large-format thresholds.

        Parameters
        ----------
        w : float
            Width of the page in inches.
        h : float
            Height of the page in inches.
        long_edge_thresh : int, optional
            Minimum longer-edge length to consider large format (default=24).
        area_thresh : int, optional
            Minimum page area in square inches to consider large format (default=800).

        Returns
        -------
        bool
            True if page is large format, False otherwise.
        """
        long_edge = max(w, h)
        area = w * h
        return long_edge >= long_edge_thresh or area >= area_thresh
    
    @staticmethod
    def pt_to_in(pt: float) -> float:
        """
        Convert points to inches.
        
        Parameters
        ----------
        pt : float
            Value in points to convert.
        
        Returns
        -------
        float
            Value in inches.
        """
        return pt / 72.0

    @property
    def pages_dims(self) -> list:
        """
        Returns the dimensions of each page in inches.
        
        Returns
        -------
        list of tuples
            A list of tuples where each tuple contains the width and height of a page in inches.
        """
        if 'pages_dims' in self.property_cache:
            return self.property_cache['pages_dims']
        logger.debug(f"Computing pages dimensions for {self.path}")
        
        with fitz.open(self.path) as doc:
            dims = [(self.pt_to_in(page.rect.width), self.pt_to_in(page.rect.height)) for page in doc]
            self.property_cache['pages_dims'] = dims
        
        return self.property_cache['pages_dims']

    @property
    def has_large_format(self) -> bool:
        """
        Determine if any page in the PDF is large format.

        Returns
        -------
        bool
            True if at least one page qualifies as large format.
        """
        #if propert_cache has the value, return it
        if 'has_large_format' in self.property_cache:
            return self.property_cache['has_large_format']
        
        logger.debug(f"Checking for large format pages in {self.path}")
        for w, h in self.pages_dims:
            if self._is_large_format_page(w, h):
                logger.debug(f"Page with size {w}x{h} inches is large format")
                self.property_cache['has_large_format'] = True
                break

        if not 'has_large_format' in self.property_cache:
            self.property_cache['has_large_format'] = False

        return self.property_cache.get('has_large_format', False)

    def pick_dpi_for_ocr(self, budget_mb: int) -> int:
        # choose worst-case page by area (inches)
        w, h = max(self.pages_dims, key=lambda wh: wh[0] * wh[1])

        budget_bytes = budget_mb * 1024 * 1024
        bytes_per_pixel = 3  # RGB
        area = w * h
        if area <= 0:
            return OCR_DPI_MAX

        dpi = int((budget_bytes / (area * bytes_per_pixel * OCR_RASTER_OVERHEAD)) ** 0.5)
        dpi = max(OCR_DPI_MIN, min(OCR_DPI_MAX, dpi))
        return dpi
    

class PDFTextExtractor(FileTextExtractor):
    """
    Extract text from PDF files with fallback to OCR.

    This class implements text extraction from PDF documents. It first attempts
    to extract text directly from the PDF. If no text is found (e.g., in scanned
    documents), it automatically falls back to OCR processing using ocrmypdf.

    Attributes
    ----------
    file_extensions : list
        Supported file extensions for this extractor.
    ocr_params : dict
        Parameters for OCR processing using ocrmypdf.
    max_stream_size : int
        Maximum file size (bytes) to process in memory before using a temp file.
    """
    # Extensions are lowercase, no leading dot (as per spec)
    file_extensions = ['pdf']

    def __init__(self):
        """
        Initialize PDFTextExtractor with default OCR parameters and stream-size threshold.
        """
        super().__init__()
        self.ocr_params = {
            'rotate_pages': True,
            'deskew': True,
            'invalidate_digital_signatures': True,
            'skip_text': True,
            'language': 'eng',
            'jobs': max(os.cpu_count() - 1, 1),  # Use all but one CPU core for OCR
            'optimize': 0,
            'output_type': 'pdf',
            'tesseract_timeout': 300,  # default timeout for Tesseract OCR
        }

        # threshold of files which cannot be processed in memory, default is 100 MB
        self.max_stream_size = 100 * 1024 * 1024

        # If OCR produces less than this many non-whitespace chars, treat as extraction failure.
        # (Allows the worker to record file_content_failures instead of silently writing junk/empty content.)
        self.ocr_min_text_chars = OCR_MIN_TEXT_CHARS
    
    @staticmethod
    def extract_text_with_ocr(pdf_path: Union[str, Path], ocr_params: dict, chunk_size: int = 0) -> str:
        """
        Perform OCR on a PDF file and return the extracted text.
        
        This method uses ocrmypdf to process PDFs that don't have extractable text,
        such as scanned documents. It creates a new PDF with an OCR text layer
        and then extracts that text.
        
        When chunk_size > 0, the PDF is processed in chunks of that many pages
        to reduce peak memory usage. This is useful for large-format documents
        that would otherwise cause out-of-memory errors during rasterization.
        
        Parameters
        ----------
        pdf_path : Union[str, Path]
            Path to the PDF file to be processed with OCR.
        ocr_params : dict
            Parameters for the OCR processing.
        chunk_size : int, optional
            Number of pages to process at once. If 0 (default), process the
            entire PDF at once. If > 0, process in chunks to reduce memory usage.

        Returns
        -------
        str
            Extracted text from the OCR-processed PDF.
            
        Raises
        ------
        FileNotFoundError
            If the input PDF file does not exist.
        """
        input_pdf_path = Path(pdf_path)
        logger.debug(f"Starting OCR extraction for {input_pdf_path} with params: {ocr_params}, chunk_size: {chunk_size}")
        if not input_pdf_path.exists():
            raise FileNotFoundError(f"Input PDF file not found for OCR operation: {input_pdf_path}")
        
        # Non-chunked processing (original behavior)
        if chunk_size <= 0:
            with tempfile.TemporaryDirectory(prefix="ocr_") as td:
                output_pdf_path = Path(td) / f"{input_pdf_path.stem}_ocr.pdf"

                params = ocr_params.copy()
                params['input_file'] = pdf_path
                params['output_file'] = output_pdf_path
                try:
                    with pil_decompression_bomb_as_error():
                        ocrmypdf.ocr(**params)
                except Exception as e:
                    if is_pil_decompression_bomb(e):
                        raise DecompressionBombTextExtractionError(
                            f"PIL decompression bomb protection triggered during OCR for {input_pdf_path}: {e}"
                        ) from e
                    raise
                logger.debug(f"OCR completed, reading text from generated PDF")

                with fitz.open(output_pdf_path) as doc:
                    return "".join(page.get_text() for page in doc)
        
        # Chunked processing to reduce peak memory usage
        logger.info(f"Processing PDF in chunks of {chunk_size} pages to reduce memory usage")
        all_text: list[str] = []
        chunks_attempted = 0
        chunks_succeeded = 0
        chunks_failed = 0
        
        with fitz.open(input_pdf_path) as src_doc:
            total_pages = src_doc.page_count
            
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size, total_pages)
                chunks_attempted += 1
                logger.debug(f"Processing pages {start_page + 1}-{end_page} of {total_pages}")
                
                with tempfile.TemporaryDirectory(prefix="ocr_chunk_") as td:
                    td_path = Path(td)
                    chunk_input = td_path / "chunk_input.pdf"
                    chunk_output = td_path / "chunk_output.pdf"
                    
                    # Extract page range to new PDF
                    with fitz.open() as chunk_doc:
                        chunk_doc.insert_pdf(src_doc, from_page=start_page, to_page=end_page - 1)
                        chunk_doc.save(chunk_input)
                    
                    # OCR the chunk
                    params = ocr_params.copy()
                    params['input_file'] = chunk_input
                    params['output_file'] = chunk_output
                    
                    try:
                        with pil_decompression_bomb_as_error():
                            ocrmypdf.ocr(**params)
                        
                        with fitz.open(chunk_output) as ocr_doc:
                            for page in ocr_doc:
                                all_text.append(page.get_text())
                        chunks_succeeded += 1
                        logger.debug(f"Successfully processed pages {start_page + 1}-{end_page}")
                    except Exception as e:
                        chunks_failed += 1
                        if is_pil_decompression_bomb(e):
                            logger.error(
                                "PIL decompression bomb protection triggered during OCR for %s pages %d-%d: %s",
                                input_pdf_path,
                                start_page + 1,
                                end_page,
                                e,
                            )
                            raise DecompressionBombTextExtractionError(
                                f"PIL decompression bomb protection triggered during OCR for {input_pdf_path} pages {start_page + 1}-{end_page}: {e}"
                            ) from e
                        logger.warning(f"OCR failed for pages {start_page + 1}-{end_page}: {e}")
                        # Continue with other chunks rather than failing entirely
                        continue

        combined = "".join(all_text)
        if not combined.strip():
            raise TextExtractionError(
                f"OCR produced no text for {input_pdf_path} (chunk_size={chunk_size}, "
                f"chunks_attempted={chunks_attempted}, chunks_succeeded={chunks_succeeded}, chunks_failed={chunks_failed})"
            )

        return combined
    
    def _fitz_doc_text(self, fitz_doc: fitz.Document, pdf_document: PDFFile) -> str:
        """
        Extract text from a fitz.Document, with fallback to OCR if any page is blank.

        Parameters
        ----------
        fitz_doc : fitz.Document
            Opened PyMuPDF document.
        pdf_document : PDFFile
            PDFFile instance for metadata and page count.

        Returns
        -------
        str
            Extracted text, using OCR if necessary.
        """
        logger.debug(f"Extracting text with fitz for document: {pdf_document.path}")
        ocr_needed_length_threshold = 100 # if found text is less than this, trigger OCR
        pdf_text = ""
        for _, page in enumerate(fitz_doc):
            page_text = page.get_text()
            pdf_text += page_text
        
        if len(pdf_text) >= ocr_needed_length_threshold:
            logger.debug(f"Extracted text length {len(pdf_text)}.")
            return pdf_text
        
        logger.info(f"OCR needed for document: {pdf_document.name}")
        ocr_params = self.ocr_params.copy()

        # Large-format pages can trigger expensive rasterization paths when doing
        # rotation/deskew. Prefer leaving pages as-is for these documents.
        if pdf_document.has_large_format:
            ocr_params["rotate_pages"] = False
            ocr_params["deskew"] = False

        # if no timeout param in ocr_params, set a default based on page count
        if not ocr_params.get('tesseract_timeout', None):
            ocr_params['tesseract_timeout'] = min(300, pdf_document.page_count * 45)

        dpi = pdf_document.pick_dpi_for_ocr(OCR_RASTER_MEM_BUDGET_MB)
        if not ocr_params.get('oversample', None):
            ocr_params['oversample'] = dpi

        if pdf_document.has_large_format and not ocr_params.get('max_image_mpixels', None):
            ocr_params['max_image_mpixels'] = OCR_MAX_IMAGE_MPIXELS_LARGE_FORMAT

        worst_w, worst_h = max(pdf_document.pages_dims, key=lambda wh: wh[0] * wh[1])
        logger.info(
            "OCR raster selection: worst_page=%.2fx%.2f in, dpi=%d, budget_mb=%d",
            worst_w,
            worst_h,
            dpi,
            OCR_RASTER_MEM_BUDGET_MB,
        )

        # set chunk_size for large-format documents
        chunk_size = 0
        if pdf_document.has_large_format or pdf_document.page_count > 20:
            chunk_size = 1 if pdf_document.has_large_format else 5

        pdf_text = self.extract_text_with_ocr(pdf_path=pdf_document.path,
                                              ocr_params=ocr_params,
                                              chunk_size=chunk_size)

        if len((pdf_text or "").strip()) < self.ocr_min_text_chars:
            raise TextExtractionError(
                f"OCR text below threshold for {pdf_document.path} "
                f"(chars={len((pdf_text or '').strip())}, min={self.ocr_min_text_chars}, "
                f"pages={pdf_document.page_count}, chunk_size={chunk_size})"
            )
        return pdf_text

    def __call__(self, pdf_filepath: str) -> str:
        """
        Extract and normalize text from the specified PDF file.

        Parameters
        ----------
        pdf_filepath : str
            Filesystem path to the PDF to process.

        Returns
        -------
        str
            Normalized extracted text.
        """
        
        # Initialize document handle and result container
        logger.debug(f"__call__: Starting extraction for file {pdf_filepath}")
        doc = None
        extracted_text = ""
        try:
            validated = validate_file(pdf_filepath)
            # Log validated path
            logger.debug(f"__call__: validated file path {validated}")
            pdf = PDFFile(validated)
            # Log PDF metadata
            logger.debug(f"__call__: PDF metadata size={pdf.size}, pages={pdf.page_count}, encrypted={pdf.is_encrypted}")

            # PyMuPDF can open encrypted PDFs only with a password; streaming doesn't help.
            if pdf.is_encrypted:
                logger.warning(f"PDF is encrypted, cannot extract text: {pdf.name}")
                raise ValueError(f"PDF file is encrypted and cannot be processed: {pdf.name}")
            
            # if the file is small enough, read it into memory
            if pdf.size <= self.max_stream_size:
                logger.debug(f"PDF size {pdf.size} <= max_stream_size ({self.max_stream_size}), processing in-memory")
                data = pdf.path.read_bytes()
                doc = fitz.open(stream=data, filetype="pdf")
                extracted_text = self._fitz_doc_text(fitz_doc=doc, pdf_document=pdf)
                doc.close()

            else:
                logger.debug(f"PDF size {pdf.size} > max_stream_size ({self.max_stream_size}), processing via temp file")
                with tempfile.TemporaryDirectory(prefix="text_extractor_") as temp_dir:
                    work_path = Path(temp_dir) / pdf.name
                    shutil.copy(pdf.path, work_path)
                    doc = fitz.open(work_path)
                    extracted_text = self._fitz_doc_text(fitz_doc=doc, pdf_document=pdf)
                    doc.close()
        
        except Exception as e:
            file_identifier = pdf.name if 'pdf' in locals() else pdf_filepath
            logger.error(f"Error extracting text from PDF {file_identifier}: {e}")
            raise

        finally:
            if doc is not None and not doc.is_closed:
                try:
                    doc.close()
                except Exception as e:
                    pass

        # Final debug before returning
        logger.debug(f"__call__: extraction complete, returning {len(extracted_text)} characters")
        return extracted_text