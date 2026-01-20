from collections.abc import Sequence
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql import func, or_

from db.models import File, FileContent
from db.db import get_db_engine
from embedding.minilm import MiniLMEmbedder
from text_extraction.pdf_extraction import PDFTextExtractor
from text_extraction.basic_extraction import TextFileTextExtractor, get_extractor_for_file


def next_file_needing_content(
    session: Session,
    *,
    cutoff: datetime,
    extensions: Sequence[str],
):
    """Return the next `File` whose content is missing or stale.

    Parameters
    ----------
    session:
        Active SQLAlchemy session bound to the archives database.
    cutoff:
        `datetime` threshold; content updated before this time is considered stale.
    extensions:
        Iterable of file extensions (with or without casing) that we know how to process.
    """
    normalized = [ext.lower() for ext in extensions or [] if ext]
    if not normalized:
        return None

    query = (
        session.query(File)
        # Load related locations for potential processing later.
        .options(selectinload(File.locations))
        # Outer join keeps files even when they have no FileContent entry yet.
        .outerjoin(FileContent, FileContent.file_hash == File.hash)
        .filter(
            # Normalize extensions so .PDF and .pdf both qualify.
            func.lower(File.extension).in_(normalized),
            # Needs processing when there is no content, no timestamp, or stale timestamp.
            or_(
                FileContent.file_hash.is_(None),
                FileContent.updated_at.is_(None),
                FileContent.updated_at < cutoff,
            ),
        )
        .order_by(File.id)
    )

    file_record = query.first()
    return file_record


def main():
    load_dotenv()
    minilm_model_name = "all-MiniLM-L6-v2"
    embedders = {
        "minilm": MiniLMEmbedder(model=minilm_model_name),
    }
    
    pdf_extractor = PDFTextExtractor()
    text_extractor = TextFileTextExtractor()
    extractors = [
        pdf_extractor,
        text_extractor,
    ]

    extensions = []
    [extensions.extend(extractor.file_extensions) for extractor in extractors]
    engine = get_db_engine()
    with Session(engine) as session:
        while True:
            file_record = next_file_needing_content(
                session,
                cutoff=datetime.utcnow(),
                extensions=extensions,
            )
            if not file_record:
                print("No more files needing content extraction.")
                break

            print(f"Processing file ID {file_record.id} at path {file_record.path}")
            extractor = get_extractor_for_file(file_record.path, extractors)
            if not extractor:
                print(f"No extractor found for file ID {file_record.id} with extension {file_record.extension}")
                continue

            try:
                text = extractor.extract_text(file_record.path)
                print(f"Extracted {len(text)} characters from file ID {file_record.id}")
                minilm_text_embeddings = embedders["minilm"].encode(text)

                # Upsert FileContent record
                content = file_record.content
                if content is None:
                    content = FileContent(file_hash=file_record.hash)
                    file_record.content = content  # establishes relationship

                content.source_text = text
                content.text_length = len(text)
                content.minilm_model = minilm_model_name
                content.minilm_emb = minilm_text_embeddings.vector
                content.updated_at = datetime.now(timezone.utc)
                session.add(content)
                session.commit()

            except Exception as e:
                print(f"Error processing file ID {file_record.id}: {e}")




if __name__ == "__main__":
    main()
