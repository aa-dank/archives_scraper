# db/models.py

import logging
import os
import fnmatch
import re
from pathlib import Path, PurePosixPath
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, BigInteger, Text, Boolean, Numeric, Index, text, Float, JSON, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

Base = declarative_base()

# get_db_engine moved to db/db.py

class File(Base):
    __tablename__ = 'files'
    id = Column(Integer, primary_key=True)
    hash = Column(String, nullable=False, unique=True, comment="SHA1 File hash for integrity checks.")
    size = Column(BigInteger, nullable=False, comment="File size in bytes.")
    extension = Column(String)
    locations = relationship("FileLocation", back_populates="file", cascade="all, delete-orphan")
    content = relationship("FileContent", back_populates="file", uselist=False, cascade="all, delete-orphan", foreign_keys="[FileContent.file_hash]")


class FileLocation(Base):
    __tablename__ = 'file_locations'
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)
    file_server_directories = Column(String)
    filename = Column(String)
    existence_confirmed = Column(DateTime(timezone=False))
    hash_confirmed = Column(DateTime(timezone=False))
    file = relationship("File", back_populates="locations")

    def local_filepath(self, server_mount_path: str) -> Path:
        if not self.file_server_directories or not self.filename:
            return None
        if not hasattr(self, '_local_path'):
            rel_parts = PurePosixPath(self.file_server_directories).parts
            self._local_path = Path(server_mount_path).joinpath(*rel_parts, self.filename)
        return self._local_path

    @property
    def file_size(self) -> int:
        file = self.file
        return file.size if file else 0


class FileContent(Base):
    """
    Stores extracted file text and vector embeddings for semantic search and ML tasks.
    """
    __tablename__ = 'file_contents'
    __table_args__ = (
        Index('ix_file_contents_minilm_emb', 'minilm_emb', postgresql_using='ivfflat', postgresql_ops={'minilm_emb': 'vector_cosine_ops'}, postgresql_with={'lists': 100}),
        Index('ix_file_contents_mpnet_emb', 'mpnet_emb', postgresql_using='ivfflat', postgresql_ops={'mpnet_emb': 'vector_cosine_ops'}, postgresql_with={'lists': 100}),
    )
    file_hash = Column(String, ForeignKey('files.hash'), primary_key=True)
    source_text = Column(Text)
    minilm_model = Column(Text, default='all-minilm-l6-v2')
    minilm_emb = Column(Vector(384))
    mpnet_model = Column(Text)
    mpnet_emb = Column(Vector(768))
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    text_length = Column(Integer)
    file = relationship("File", back_populates="content", foreign_keys=[file_hash])


class ArchivedFile(Base):
    __tablename__ = 'archived_files'
    id = Column(Integer, primary_key=True)
    destination_path = Column(String, nullable=False)
    project_number = Column(String)
    document_date = Column(String)
    destination_directory = Column(String)
    file_code = Column(String)
    file_size = Column(Float, nullable=False)
    date_archived = Column(DateTime(timezone=False), nullable=False)
    archivist_id = Column(Integer, nullable=False)
    notes = Column(String)
    filename = Column(String)
    file_id = Column(Integer)


class Caan(Base):
    __tablename__ = 'caans'
    id = Column(Integer, primary_key=True)
    caan = Column(String, nullable=False)
    name = Column(String)
    description = Column(String)
    project_caans = relationship("ProjectCaan", back_populates="caan")


class Project(Base):
    __tablename__ = 'projects'
    id = Column(Integer, primary_key=True)
    number = Column(String, nullable=False)
    name = Column(String, nullable=False)
    file_server_location = Column(String)
    drawings = Column(Boolean)
    project_caans = relationship("ProjectCaan", back_populates="project")


class ProjectCaan(Base):
    __tablename__ = 'project_caans'
    project_id = Column(Integer, ForeignKey('projects.id'), primary_key=True)
    caan_id = Column(Integer, ForeignKey('caans.id'), primary_key=True)
    project = relationship("Project", back_populates="project_caans")
    caan = relationship("Caan", back_populates="project_caans")


class SchemaMigration(Base):
    __tablename__ = 'schema_migration'
    onerow_id = Column(Boolean, primary_key=True, default=True)
    background = Column(Integer, nullable=False)
    data_loaded = Column(Boolean, nullable=False)
    foreground = Column(Integer, nullable=False)


class ServerChange(Base):
    __tablename__ = 'server_changes'
    id = Column(Integer, primary_key=True)
    old_path = Column(String)
    new_path = Column(String)
    change_type = Column(String, nullable=False)
    files_effected = Column(Integer)
    data_effected = Column(Numeric)
    date = Column(DateTime(timezone=False), nullable=False)
    user_id = Column(Integer, nullable=False)


class Timekeeper(Base):
    __tablename__ = 'timekeeper'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    datetime = Column(DateTime(timezone=False), nullable=False)
    clock_in_event = Column(Boolean, nullable=False)
    journal = Column(Text)


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String, nullable=False, unique=True)
    first_name = Column(String)
    last_name = Column(String)
    roles = Column(String)
    password = Column(String(60))
    active = Column(Boolean, nullable=False)


class WorkerTask(Base):
    __tablename__ = 'worker_tasks'
    id = Column(Integer, primary_key=True)
    task_id = Column(String(255), nullable=False)
    time_enqueued = Column(DateTime(timezone=False), nullable=False)
    time_completed = Column(DateTime(timezone=False))
    origin = Column(String(255), nullable=False)
    function_name = Column(String(255))
    status = Column(String(255))
    task_results = Column(JSON)


class FileContentFailure(Base):
    __tablename__ = "file_content_failures"
    __table_args__ = (
        CheckConstraint(
            "stage in ('extract', 'embed')",
            name="file_content_failures_stage_check",
        ),
    )
    file_hash = Column(
        String,
        ForeignKey("files.hash", ondelete="CASCADE"),
        primary_key=True,
        comment="Hash of the file that failed processing. One active failure record per file.",
    )
    stage = Column(
        String(16),
        nullable=False,
        comment="Processing stage that failed: extract (text extraction) or embed (embedding generation).",
    )
    error = Column(
        Text,
        comment="Human-readable error message or exception summary for the most recent failure.",
    )
    attempts = Column(
        Integer,
        nullable=False,
        server_default=text("1"),
        comment="Number of failed processing attempts recorded for this file.",
    )
    last_failed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Timestamp of the most recent failure occurrence.",
    )
    file = relationship(
        "File",
        foreign_keys=[file_hash],
        passive_deletes=True,
    )