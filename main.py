from collections.abc import Sequence
from datetime import datetime
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql import func, or_

from db.models import File, FileContent

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
    print("Hello from archives-scraper!")


if __name__ == "__main__":
    main()
