# db/models.py

import logging
import os
import fnmatch
import re
from pathlib import Path, PurePosixPath
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, BigInteger, Text, Boolean, Numeric, Index, Date, or_
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
    size = Column(BigInteger, nullable=False, comment="File size in bytes.")
    hash = Column(String, nullable=False, unique=True, comment="SHA1 File hash for integrity checks.")
    extension = Column(String)
    locations = relationship("FileLocation", back_populates="file", cascade="all, delete-orphan")
    tag_labels = relationship("FileTagLabel", back_populates="file", cascade="all, delete-orphan", foreign_keys="[FileTagLabel.file_hash]")
    content = relationship("FileContent", back_populates="file", uselist=False, cascade="all, delete-orphan")
    collection_members = relationship("FileCollectionMember", back_populates="file", cascade="all, delete-orphan", foreign_keys="[FileCollectionMember.file_id]")

    date_mentions = relationship(
        "FileDateMention",
        back_populates="file",
        cascade="all, delete-orphan",
        foreign_keys="[FileDateMention.file_hash]"
    )


class FileLocation(Base):
    __tablename__ = 'file_locations'
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)
    existence_confirmed = Column(DateTime(timezone=True))
    hash_confirmed = Column(DateTime(timezone=True))
    file_server_directories = Column(String)
    filename = Column(String)
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


class FilingTag(Base):
    __tablename__ = 'filing_tags'
    label = Column(Text, primary_key=True)
    parent_label = Column(Text, ForeignKey('filing_tags.label'))
    description = Column(Text)
    importance_rank = Column(Integer)
    confidence_floor = Column(Numeric, default=0.60)
    parent = relationship("FilingTag", remote_side=[label], back_populates="children")
    children = relationship("FilingTag", back_populates="parent")
    file_labels = relationship("FileTagLabel", back_populates="filing_tag")

    @property
    def full_tag_label_str(self) -> str:
        return f"{self.label} - {self.description}".strip()

    @property
    def label_search_str(self) -> str:
        return f"{self.label} - "

    @classmethod
    def retrieve_tag_by_label(cls, session, label_str: str) -> 'FilingTag':
        if ' ' in label_str:
            label_str = label_str.split(' ')[0]
        return session.query(cls).filter_by(label=label_str).first()


class FileTagLabel(Base):
    __tablename__ = 'file_tag_labels'
    file_id = Column(Integer, ForeignKey('files.id'), primary_key=True)
    file_hash = Column(String, ForeignKey('files.hash'), nullable=False)
    tag = Column(Text, ForeignKey('filing_tags.label'), primary_key=True)
    is_primary = Column(Boolean, default=True)
    label_source = Column(Text, default='human')
    split = Column(Text, default='train')
    file = relationship("File", back_populates="tag_labels", foreign_keys=[file_hash])
    filing_tag = relationship("FilingTag", back_populates="file_labels")


class FileContent(Base):
    """
    Stores extracted file text and vector embeddings for semantic search and ML tasks.
    Previously named FileEmbedding (renamed for clarity).
    """
    __tablename__ = 'file_contents'
    __table_args__ = (
        Index('ix_file_contents_minilm_emb', 'minilm_emb', postgresql_using='ivfflat', postgresql_ops={'minilm_emb': 'vector_cosine_ops'}, postgresql_with={'lists': 100}),
        Index('ix_file_contents_mpnet_emb', 'mpnet_emb', postgresql_using='ivfflat', postgresql_ops={'mpnet_emb': 'vector_cosine_ops'}, postgresql_with={'lists': 100}),
    )
    file_hash = Column(String, ForeignKey('files.hash'), primary_key=True)
    source_text = Column(Text)
    text_length = Column(Integer, comment="Length of the extracted text in characters.")
    minilm_model = Column(Text)
    minilm_emb = Column(Vector(384))
    mpnet_model = Column(Text)
    mpnet_emb = Column(Vector(768))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    file = relationship("File", back_populates="content", foreign_keys=[file_hash])


class FileCollection(Base):
    __tablename__ = 'file_collections'
    id          = Column(Integer, primary_key=True)
    name        = Column(Text, nullable=False, unique=True)
    description = Column(Text, comment="Human-readable description of the collection.", nullable=True)
    meta    = Column('metadata', JSONB, nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    members = relationship(
        "FileCollectionMember",
        back_populates="collection",
        cascade="all, delete-orphan"
    )


class FileCollectionMember(Base):
    __tablename__ = 'file_collection_members'
    collection_id = Column(Integer, ForeignKey('file_collections.id', ondelete='CASCADE'), primary_key=True)
    file_id       = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), primary_key=True)
    role          = Column(Text, nullable=False, comment="Role of this file in the collection: 'train', 'test', 'val', 'prototype', etc.")
    added_at      = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    collection = relationship(
        "FileCollection",
        back_populates="members"
    )
    file = relationship(
        "File",
        back_populates="collection_members"
    )


class FileDateMention(Base):
    __tablename__ = 'file_date_mentions'
    __table_args__ = (
        Index('ix_file_date_mentions_date', 'mention_date'),
        Index('ix_file_date_mentions_file', 'file_hash'),
        Index('ix_file_date_mentions_date_gran', 'mention_date', 'granularity'),
    )

    # Link to files via the unique file hash (consistent with FileContent / FileTagLabel)
    file_hash     = Column(String, ForeignKey('files.hash'), primary_key=True)
    mention_date  = Column(Date, primary_key=True)           # normalized calendar date
    granularity   = Column(Text, primary_key=True, default='day')  # 'day' | 'month' | 'year'
    mentions_count= Column(Integer, nullable=False, default=1)     # per-file count for this date
    extractor     = Column(Text, nullable=True)              # e.g., 'dateparser@1.2.3'
    extracted_at  = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    file = relationship("File", back_populates="date_mentions", foreign_keys=[file_hash])


class PathPattern(Base):
    __tablename__ = 'path_patterns'
    id           = Column(Integer,   primary_key=True)
    pattern      = Column(String,    nullable=False, unique=True)
    pattern_type = Column(String,    nullable=False)  # 'directory','file','regex'
    description  = Column(Text,      nullable=True)
    treatment    = Column(String,    nullable=False)  # 'exclude','priority',...
    meta         = Column('metadata', JSONB,         nullable=True)
    contexts     = Column(JSONB,     nullable=True)   # e.g. ['add_files','date_mentions']
    enabled      = Column(Boolean,   default=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at   = Column(DateTime(timezone=True),
                           server_default=func.now(),
                           onupdate=func.now())

    @classmethod
    def get_active_patterns(cls, session, treatment=None, context=None):
        """Return only enabled patterns, filtered by treatment and (optional) context."""
        q = session.query(cls).filter_by(enabled=True)
        if treatment:
            q = q.filter(cls.treatment == treatment)
        if context:
            # include patterns with no contexts (global) or that list this context
            q = q.filter(
                or_(
                  cls.contexts == None,
                  cls.contexts.contains([context])
                )
            )
        rows = q.all()

        out = {'directory': [], 'file': [], 'regex': []}
        for r in rows:
            t = r.pattern_type.lower()
            if t in out:
                out[t].append({'pattern': r.pattern})
        return out

    @classmethod
    def is_excluded(cls,
                    session,
                    path: str,
                    context: str | None = None) -> bool:
        """Skip if any enabled ‘exclude’ pattern matches for this context."""
        import fnmatch, re, os
        path = path.replace('\\','/')
        name = os.path.basename(path)

        pats = cls.get_active_patterns(session,
                                       treatment='exclude',
                                       context=context)
        for d in pats['directory']:
            if fnmatch.fnmatch(path, d['pattern']):
                return True
        for f in pats['file']:
            if fnmatch.fnmatch(name, f['pattern']):
                return True
        for r in pats['regex']:
            try:
                if re.search(r['pattern'], path):
                    return True
            except re.error:
                logger.warning(f"bad regex {r['pattern']}")
        return False
    
    @classmethod
    def check_path_treatment(cls, session, path: str):
        """
        Check all treatments that apply to a path.
        
        Args:
            session: Database session
            path: Path to check
            
        Returns:
            Dict of applicable treatments with matching pattern details
        """
        
        path = path.replace('\\', '/')
        filename = os.path.basename(path)
        
        # Get all active patterns
        all_patterns = cls.get_active_patterns(session)
        matched_treatments = {}
        
        # Check directory patterns
        for p in all_patterns.get("directory", []):
            if fnmatch.fnmatch(path, p['pattern']):
                treatment = p['treatment']
                if treatment not in matched_treatments:
                    matched_treatments[treatment] = []
                matched_treatments[treatment].append(p)
        
        # Check file patterns
        for p in all_patterns.get("file", []):
            if fnmatch.fnmatch(filename, p['pattern']):
                treatment = p['treatment']
                if treatment not in matched_treatments:
                    matched_treatments[treatment] = []
                matched_treatments[treatment].append(p)
        
        # Check regex patterns
        for p in all_patterns.get("regex", []):
            try:
                if re.search(p['pattern'], path):
                    treatment = p['treatment']
                    if treatment not in matched_treatments:
                        matched_treatments[treatment] = []
                    matched_treatments[treatment].append(p)
            except re.error:
                logger.warning(f"Invalid regex pattern: {p['pattern']}")
                continue
                
        return matched_treatments