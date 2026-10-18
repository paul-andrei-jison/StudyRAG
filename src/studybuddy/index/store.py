"""SQLite store: documents, pages, chunks and the FTS5 keyword index.

SQLite is the source of truth for all text. Vectors (LanceDB) can be rebuilt from it at any time.
`image_path` is the original image file; for a PDF it is the PDF itself and `page_number` selects the page.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..errors import IndexStoreError

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    subject_key TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    page_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    text TEXT NOT NULL,
    quality REAL,
    flags TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    record_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (doc_id, page_number)
);
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    image_path TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    subject_key TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT ''
);
CREATE INDEX chunks_doc ON chunks(doc_id);
CREATE INDEX chunks_subject ON chunks(subject_key);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, title, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, title) VALUES (new.id, new.text, new.title);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, title) VALUES ('delete', old.id, old.text, old.title);
END;
CREATE TRIGGER chunks_au AFTER UPDATE OF text, title ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, title) VALUES ('delete', old.id, old.text, old.title);
    INSERT INTO chunks_fts(rowid, text, title) VALUES (new.id, new.text, new.title);
END;
"""


def subject_key(subject: str | None) -> str:
    """Case-insensitive key used for subject filters."""
    return (subject or "").strip().casefold()


@dataclass
class DocumentMeta:
    content_hash: str
    file_path: str
    source_kind: str            # photo | photo-file | pdf | document | text-file
    subject: str = ""
    title: str = ""
    summary: str = ""


@dataclass
class PageData:
    page_number: int
    image_path: str
    source_kind: str
    text: str
    quality: float | None = None
    flags: list[str] = field(default_factory=list)
    error: str | None = None
    record_json: str = "{}"


@dataclass
class ChunkData:
    page_number: int
    chunk_index: int
    text: str


@dataclass
class ChunkRow:
    id: int
    doc_id: int
    page_number: int
    chunk_index: int
    text: str
    image_path: str
    subject: str
    title: str
    file_path: str
    page_quality: float | None
    page_flags: list[str]


@dataclass
class DocumentRow:
    id: int
    content_hash: str
    file_path: str
    source_kind: str
    subject: str
    title: str
    summary: str
    page_count: int
    chunk_count: int
    created_at: str


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            self._create_schema()
        elif version != SCHEMA_VERSION:
            raise IndexStoreError(f"{db_path} has schema version {version}, expected {SCHEMA_VERSION}")

    def _create_schema(self) -> None:
        with self._lock:
            try:
                self._db.executescript(f"BEGIN;\n{SCHEMA}\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;")
            except Exception:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            else:
                self._db.execute("COMMIT")

    # ---- writes ---------------------------------------------------------------------------

    def find_by_hash(self, content_hash: str) -> int | None:
        with self._lock:
            row = self._db.execute("SELECT id FROM documents WHERE content_hash = ?", (content_hash,)).fetchone()
        return row["id"] if row else None

    def add_document(self, meta: DocumentMeta, pages: list[PageData], chunks: list[ChunkData]
                     ) -> tuple[int, list[int], bool]:
        """Insert a document with ALL its pages and chunks atomically.

        Returns (doc_id, chunk_ids, created). If the content hash already exists nothing is written and
        (existing_id, [], False) is returned. Chunk ids are returned in `chunks` order.
        """
        numbers = [p.page_number for p in pages]
        if len(set(numbers)) != len(numbers):
            raise IndexStoreError(f"Duplicate page numbers: {numbers}")
        unknown = {c.page_number for c in chunks} - set(numbers)
        if unknown:
            raise IndexStoreError(f"Chunks refer to pages that were not given: {sorted(unknown)}")
        path_of = {p.page_number: p.image_path for p in pages}
        key = subject_key(meta.subject)
        with self.transaction() as db:
            existing = db.execute("SELECT id FROM documents WHERE content_hash = ?", (meta.content_hash,)).fetchone()
            if existing:
                return existing["id"], [], False
            doc_id = db.execute(
                "INSERT INTO documents (content_hash, file_path, source_kind, subject, subject_key, title, summary,"
                " page_count, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (meta.content_hash, meta.file_path, meta.source_kind, meta.subject, key, meta.title, meta.summary,
                 len(pages), datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ).lastrowid
            for p in pages:
                db.execute(
                    "INSERT INTO pages (doc_id, page_number, image_path, source_kind, text, quality, flags, error,"
                    " record_json) VALUES (?,?,?,?,?,?,?,?,?)",
                    (doc_id, p.page_number, p.image_path, p.source_kind, p.text, p.quality,
                     json.dumps(p.flags), p.error, p.record_json),
                )
            chunk_ids = [
                db.execute(
                    "INSERT INTO chunks (doc_id, page_number, chunk_index, text, image_path, subject, subject_key,"
                    " title) VALUES (?,?,?,?,?,?,?,?)",
                    (doc_id, c.page_number, c.chunk_index, c.text, path_of[c.page_number], meta.subject, key,
                     meta.title),
                ).lastrowid
                for c in chunks
            ]
            stored = db.execute("SELECT count(*) FROM pages WHERE doc_id = ?", (doc_id,)).fetchone()[0]
            if stored != len(pages):
                raise IndexStoreError(f"Stored {stored} pages but {len(pages)} were given")
        return doc_id, chunk_ids, True

    def delete_document(self, doc_id: int) -> list[int]:
        """Delete a document with its pages and chunks; returns the removed chunk ids."""
        with self.transaction() as db:
            ids = [r["id"] for r in db.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
            db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))  # cascades to pages and chunks
        return ids

    def update_document(self, doc_id: int, *, subject: str | None = None, title: str | None = None,
                        summary: str | None = None, file_path: str | None = None) -> list[int]:
        """Change document fields; subject and title are copied onto its chunks. Returns the chunk ids."""
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone() is None:
                raise IndexStoreError(f"No document with id {doc_id}")
            if subject is not None:
                db.execute("UPDATE documents SET subject = ?, subject_key = ? WHERE id = ?",
                           (subject, subject_key(subject), doc_id))
                db.execute("UPDATE chunks SET subject = ?, subject_key = ? WHERE doc_id = ?",
                           (subject, subject_key(subject), doc_id))
            if title is not None:
                db.execute("UPDATE documents SET title = ? WHERE id = ?", (title, doc_id))
                db.execute("UPDATE chunks SET title = ? WHERE doc_id = ?", (title, doc_id))
            if summary is not None:
                db.execute("UPDATE documents SET summary = ? WHERE id = ?", (summary, doc_id))
            if file_path is not None:
                db.execute("UPDATE documents SET file_path = ? WHERE id = ?", (file_path, doc_id))
            return [r["id"] for r in db.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]

    def rewrite_paths(self, doc_id: int, subject: str, mapping: dict[str, str]) -> None:
        """Give a document another subject and point every stored path at its new place, in ONE transaction.

        `mapping` is old path -> new path (the document's file or folder and each page file). Covers documents.file_path,
        pages.image_path, the source_path inside pages.record_json and chunks.image_path; nothing changes if any step fails."""
        with self.transaction() as db:
            doc = db.execute("SELECT file_path FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if doc is None:
                raise IndexStoreError(f"No document with id {doc_id}")
            key = subject_key(subject)
            db.execute("UPDATE documents SET subject = ?, subject_key = ?, file_path = ? WHERE id = ?",
                       (subject, key, mapping.get(doc["file_path"], doc["file_path"]), doc_id))
            db.execute("UPDATE chunks SET subject = ?, subject_key = ? WHERE doc_id = ?", (subject, key, doc_id))
            for page in db.execute("SELECT id, image_path, record_json FROM pages WHERE doc_id = ?", (doc_id,)).fetchall():
                try:
                    record = json.loads(page["record_json"] or "{}")
                except ValueError:
                    record = {}
                if isinstance(record, dict) and record.get("source_path") in mapping:
                    record["source_path"] = mapping[record["source_path"]]
                db.execute("UPDATE pages SET image_path = ?, record_json = ? WHERE id = ?",
                           (mapping.get(page["image_path"], page["image_path"]), json.dumps(record, default=str), page["id"]))
            for old, new in mapping.items():
                db.execute("UPDATE chunks SET image_path = ? WHERE doc_id = ? AND image_path = ?", (new, doc_id, old))

    # ---- reads ----------------------------------------------------------------------------

    def documents(self) -> list[DocumentRow]:
        with self._lock:
            rows = self._db.execute(
                "SELECT d.*, (SELECT count(*) FROM chunks c WHERE c.doc_id = d.id) AS chunk_count "
                "FROM documents d ORDER BY d.id"
            ).fetchall()
        return [DocumentRow(r["id"], r["content_hash"], r["file_path"], r["source_kind"], r["subject"], r["title"],
                            r["summary"], r["page_count"], r["chunk_count"], r["created_at"]) for r in rows]

    def get_document(self, doc_id: int) -> DocumentRow | None:
        return next((d for d in self.documents() if d.id == doc_id), None)

    def pages(self, doc_id: int) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute("SELECT * FROM pages WHERE doc_id = ? ORDER BY page_number", (doc_id,)).fetchall()

    def chunks_by_id(self, ids: list[int]) -> dict[int, ChunkRow]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._lock:
            rows = self._db.execute(
                f"SELECT c.*, d.file_path AS file_path, p.quality AS page_quality, p.flags AS page_flags "
                f"FROM chunks c JOIN documents d ON d.id = c.doc_id "
                f"LEFT JOIN pages p ON p.doc_id = c.doc_id AND p.page_number = c.page_number "
                f"WHERE c.id IN ({marks})", ids,
            ).fetchall()
        return {r["id"]: ChunkRow(r["id"], r["doc_id"], r["page_number"], r["chunk_index"], r["text"], r["image_path"],
                                  r["subject"], r["title"], r["file_path"], r["page_quality"],
                                  json.loads(r["page_flags"] or "[]")) for r in rows}

    def all_chunks(self) -> list[tuple[int, str, str, int]]:
        """(chunk id, text, subject_key, doc id) for every chunk, in id order."""
        with self._lock:
            rows = self._db.execute("SELECT id, text, subject_key, doc_id FROM chunks ORDER BY id").fetchall()
        return [(r["id"], r["text"], r["subject_key"], r["doc_id"]) for r in rows]

    def chunk_ids(self) -> set[int]:
        with self._lock:
            return {r[0] for r in self._db.execute("SELECT id FROM chunks")}

    def keyword_search(self, match: str, limit: int, subject: str | None = None) -> list[tuple[int, float]]:
        """FTS5 search. `match` must already be a safe FTS5 query. Returns (chunk id, score), best first;
        score is -bm25 so bigger is better."""
        sql = ("SELECT c.id AS id, bm25(chunks_fts, 1.0, 0.3) AS s FROM chunks_fts "
               "JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ?")
        params: list = [match]
        if subject:
            sql += " AND c.subject_key = ?"
            params.append(subject_key(subject))
        sql += " ORDER BY s LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        return [(r["id"], -r["s"]) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {t: self._db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                    for t in ("documents", "pages", "chunks")}
