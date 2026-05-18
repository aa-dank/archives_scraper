"""
Text chunking utilities for FTS (Full-Text Search) indexing.

Provides simple, reusable text chunking logic for both archives_scraper and desktop_archives_scraper.
Chunks are designed for database persistence in file_content_fts_chunks table.
"""

from datetime import datetime, timezone
from typing import Iterable


class TextChunker:
    """
    Break extracted text into fixed-size chunks for FTS indexing.
    
    Chunks are overlapping-aware; empty chunks are skipped.
    """
    
    DEFAULT_CHUNK_SIZE = 20_000  # Characters, matches database backfill
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Iterable[tuple[int, str]]:
        """
        Break text into fixed-size chunks.
        
        Yields (chunk_index, chunk_text) tuples for non-empty chunks.
        Empty or whitespace-only chunks are skipped.
        
        Parameters
        ----------
        text : str
            Source text to chunk.
        chunk_size : int, optional
            Target characters per chunk (default 20_000).
            
        Yields
        ------
        tuple[int, str]
            (chunk_index, chunk_text) where chunk_index is 0-based.
        """
        if not text or not text.strip():
            return
        
        text_len = len(text)
        chunk_index = 0
        start = 0
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]
            
            # Skip empty chunks
            if chunk_text.strip():
                yield chunk_index, chunk_text
            
            start = end
            chunk_index += 1
    
    @staticmethod
    def build_chunk_rows(
        file_hash: str,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunked_at: datetime | None = None,
    ) -> list[dict]:
        """
        Generate FileContentFtsChunk row dictionaries ready for insertion.
        
        Each row dict maps to file_content_fts_chunks columns:
        - file_hash
        - chunk_index
        - chunk_text
        - chunked_at
        
        Parameters
        ----------
        file_hash : str
            SHA1 hash of the source file.
        text : str
            Full source text to chunk.
        chunk_size : int, optional
            Target characters per chunk (default 20_000).
        chunked_at : datetime | None, optional
            Timestamp for chunking operation. Defaults to now() in UTC.
            
        Returns
        -------
        list[dict]
            List of row dicts, empty if text is empty or whitespace-only.
        """
        if chunked_at is None:
            chunked_at = datetime.now(timezone.utc)
        
        rows = [
            {
                'file_hash': file_hash,
                'chunk_index': idx,
                'chunk_text': chunk_text,
                'chunked_at': chunked_at,
            }
            for idx, chunk_text in TextChunker.chunk_text(text, chunk_size)
        ]
        return rows
