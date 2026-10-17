"""The Index facade: guarded writes across SQLite and LanceDB, consistency checks, re-index and reset.

Write order for a document: chunk, embed (nothing is written yet), SQLite transaction, add vectors; if the
vector write fails the SQLite rows are removed again. A crash between the two stores can leave orphan vectors
or chunks without a vector; `check()` finds them and `repair()` fixes them.
"""
from __future__ import annotations

import functools
import json
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

from ..config import Settings
from ..embedding_guard import META_FILENAME, check_embedding, read_meta, write_meta
from ..errors import IndexMismatchError, IndexStoreError
from ..reader.quality import PageRecord
from .chunking import chunk_page
from .store import ChunkData, DocumentMeta, DocumentRow, PageData, Store, subject_key
from .vectors import VectorStore

REINDEX_BATCH = 32
INDEX_DIRNAME = "index"
DB_NAME = "studybuddy.db"
VECTORS_NAME = "vectors"
REINDEX_HINT = "Run: python scripts/index_admin.py reindex"


def _serialized(method):
    """Run a write method under the index's write lock. The ingest service already funnels writes through one
    worker; this keeps the index consistent even if some other code path writes too."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._write_lock:
            return method(self, *args, **kwargs)
    return wrapper


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class AddResult:
    doc_id: int
    created: bool          # False: the same content hash was already indexed, nothing was written
    chunk_count: int


@dataclass
class Consistency:
    missing_vectors: set[int] = field(default_factory=set)   # chunks in SQLite without a vector
    orphan_vectors: set[int] = field(default_factory=set)    # vectors without a chunk in SQLite

    @property
    def ok(self) -> bool:
        return not self.missing_vectors and not self.orphan_vectors


@dataclass
class Status:
    documents: int
    pages: int
    chunks: int
    vectors: int
    dim: int | None
    recorded_model: str | None
    recorded_doc_prefix: str | None
    configured_model: str
    configured_doc_prefix: str
    consistency: Consistency
    guard_problem: str | None = None


def index_dir(settings: Settings) -> Path:
    return settings.data_dir / INDEX_DIRNAME


def reset_index(settings: Settings) -> list[Path]:
    """Delete the whole index (SQLite files, vectors, embedding record). Never touches the library folder."""
    removed: list[Path] = []
    root = index_dir(settings)
    if root.exists():
        removed.append(root)
        shutil.rmtree(root)
    meta = settings.data_dir / META_FILENAME
    if meta.exists():
        meta.unlink()
        removed.append(meta)
    return removed


class Index:
    def __init__(self, settings: Settings, embedder: Embedder, store: Store, vectors: VectorStore):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.vectors = vectors
        self._write_lock = threading.RLock()

    @classmethod
    def open(cls, settings: Settings, embedder: Embedder, check_guard: bool = True) -> "Index":
        root = index_dir(settings)
        store = Store(root / DB_NAME)
        vectors = VectorStore(root / VECTORS_NAME)
        index = cls(settings, embedder, store, vectors)
        if check_guard:
            index._verify_guard()
        return index

    def close(self) -> None:
        self.store.close()

    # ---- embedding guard ------------------------------------------------------------------

    def _is_empty(self) -> bool:
        return self.store.counts()["chunks"] == 0 and self.vectors.count() == 0

    def _verify_guard(self, dim: int | None = None) -> None:
        """Refuse to mix vectors from another model/prefix. An empty index has nothing to protect."""
        data_dir = self.settings.data_dir
        if self._is_empty() and read_meta_safe(data_dir) is not None and dim is None:
            (data_dir / META_FILENAME).unlink(missing_ok=True)  # stale record, e.g. from check_models.py
        dim = dim or self.vectors.dim
        if dim is None:
            return  # nothing embedded yet: the first add records the model
        result = check_embedding(data_dir, self.settings.embed_model, dim, self.settings.embed_doc_prefix)
        if not result.ok:
            raise IndexMismatchError(f"{result.message} {REINDEX_HINT}")

    # ---- writes ---------------------------------------------------------------------------

    @_serialized
    def add_document(self, meta: DocumentMeta, records: Sequence[PageRecord],
                     expected_pages: int | None = None) -> AddResult:
        """Index a document. Every record becomes a page row; pages with text become chunks.

        `expected_pages` is the number of pages that went into the reader: if it differs from the records
        given, nothing is written (pages in must equal pages out).
        """
        if expected_pages is not None and expected_pages != len(records):
            raise IndexStoreError(f"Page mismatch: expected {expected_pages} pages, got {len(records)} records")
        existing = self.store.find_by_hash(meta.content_hash)
        if existing is not None:
            return AddResult(existing, False, 0)

        pages = [PageData(r.page_number, r.source_path, r.source_kind, r.text, r.quality, list(r.flags), r.error,
                          json.dumps(asdict(r), default=str)) for r in records]
        chunks: list[ChunkData] = []
        for r in records:
            if r.text.strip() and not r.error:
                chunks.extend(ChunkData(c.page_number, c.chunk_index, c.text) for c in chunk_page(r.text, r.page_number))

        vectors: list[list[float]] = []
        if chunks:  # embed BEFORE writing anything: an embedding failure leaves no trace
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            if len(vectors) != len(chunks):
                raise IndexStoreError(f"Embedded {len(vectors)} vectors for {len(chunks)} chunks")
            self._verify_guard(dim=len(vectors[0]))

        doc_id, chunk_ids, created = self.store.add_document(meta, pages, chunks)
        if not created:
            return AddResult(doc_id, False, 0)
        try:
            key = subject_key(meta.subject)
            self.vectors.add(chunk_ids, vectors, [key] * len(chunk_ids), [doc_id] * len(chunk_ids))
        except BaseException:
            self.store.delete_document(doc_id)
            self.vectors.delete(chunk_ids)
            raise
        return AddResult(doc_id, True, len(chunk_ids))

    @_serialized
    def delete_document(self, doc_id: int) -> int:
        ids = self.store.delete_document(doc_id)
        self.vectors.delete(ids)
        return len(ids)

    @_serialized
    def update_document(self, doc_id: int, *, subject: str | None = None, title: str | None = None,
                        summary: str | None = None, file_path: str | None = None) -> None:
        self.store.update_document(doc_id, subject=subject, title=title, summary=summary, file_path=file_path)
        if subject is not None:
            self.vectors.update_subject(doc_id, subject_key(subject))

    def set_subject(self, doc_id: int, subject: str) -> None:
        self.update_document(doc_id, subject=subject)

    # ---- reads ----------------------------------------------------------------------------

    def documents(self) -> list[DocumentRow]:
        return self.store.documents()

    def search(self, query: str, k: int = 8, subject: str | None = None, mode: str = "hybrid"):
        from .search import search  # local import: search depends on this module's types

        return search(self, query, k=k, subject=subject, mode=mode)

    # ---- health ---------------------------------------------------------------------------

    def check(self) -> Consistency:
        have, stored = self.store.chunk_ids(), self.vectors.ids()
        return Consistency(missing_vectors=have - stored, orphan_vectors=stored - have)

    @_serialized
    def repair(self) -> Consistency:
        """Delete orphan vectors and embed chunks that lack one. Returns what was fixed."""
        found = self.check()
        if found.orphan_vectors:
            self.vectors.delete(sorted(found.orphan_vectors))
        if found.missing_vectors:
            wanted = [c for c in self.store.all_chunks() if c[0] in found.missing_vectors]
            vectors = self.embedder.embed_documents([c[1] for c in wanted])
            self._verify_guard(dim=len(vectors[0]))
            self.vectors.add([c[0] for c in wanted], vectors, [c[2] for c in wanted], [c[3] for c in wanted])
        return found

    def status(self) -> Status:
        counts = self.store.counts()
        recorded = read_meta_safe(self.settings.data_dir)
        problem = None
        try:
            if self.vectors.dim is not None:
                result = check_embedding_readonly(self.settings, self.vectors.dim)
                problem = None if result is None else result
        except Exception as exc:  # status must never crash the diagnosis
            problem = str(exc)
        return Status(counts["documents"], counts["pages"], counts["chunks"], self.vectors.count(), self.vectors.dim,
                      recorded["model"] if recorded else None, recorded.get("doc_prefix") if recorded else None,
                      self.settings.embed_model, self.settings.embed_doc_prefix, self.check(), problem)

    @_serialized
    def reindex(self, progress: Callable[[int, int], None] | None = None) -> int:
        """Re-embed every chunk from SQLite with the CURRENT model and document prefix.

        Vectors go into a fresh table that replaces the old one only when every chunk is done; any failure
        leaves the old vectors and the embedding record untouched. Returns the number of chunks embedded.
        """
        root = index_dir(self.settings)
        new_path, old_path, live_path = root / "vectors.new", root / "vectors.old", root / VECTORS_NAME
        for leftover in (new_path, old_path):
            if leftover.exists():
                shutil.rmtree(leftover)
        chunks = self.store.all_chunks()
        fresh = None
        dim: int | None = None
        try:
            for start in range(0, len(chunks), REINDEX_BATCH):
                batch = chunks[start:start + REINDEX_BATCH]
                vectors = self.embedder.embed_documents([c[1] for c in batch])
                if len(vectors) != len(batch):
                    raise IndexStoreError(f"Embedded {len(vectors)} vectors for {len(batch)} chunks")
                if fresh is None:
                    dim = len(vectors[0])
                    fresh = VectorStore(new_path, dim=dim)
                fresh.add([c[0] for c in batch], vectors, [c[2] for c in batch], [c[3] for c in batch])
                if progress:
                    progress(min(start + REINDEX_BATCH, len(chunks)), len(chunks))
            if fresh is not None and fresh.count() != len(chunks):
                raise IndexStoreError(f"New vector table has {fresh.count()} rows for {len(chunks)} chunks")
        except BaseException:
            fresh = None
            shutil.rmtree(new_path, ignore_errors=True)
            raise

        fresh = None  # release the new table before moving directories (Windows file locks)
        self.vectors = None  # type: ignore[assignment]
        try:
            if live_path.exists():
                live_path.rename(old_path)
            if chunks:
                new_path.rename(live_path)
            shutil.rmtree(old_path, ignore_errors=True)
        except BaseException:
            if old_path.exists() and not live_path.exists():
                old_path.rename(live_path)
            raise
        finally:
            self.vectors = VectorStore(live_path)
        if chunks:
            write_meta(self.settings.data_dir, self.settings.embed_model, dim, self.settings.embed_doc_prefix)
        else:
            (self.settings.data_dir / META_FILENAME).unlink(missing_ok=True)
        return len(chunks)


def read_meta_safe(data_dir: Path) -> dict | None:
    try:
        return read_meta(data_dir)
    except (ValueError, KeyError, TypeError):
        return None


def check_embedding_readonly(settings: Settings, dim: int) -> str | None:
    """Like the guard, but never writes. Returns a problem description or None."""
    meta = read_meta_safe(settings.data_dir)
    if meta is None:
        return None
    if meta["model"] != settings.embed_model or meta["dim"] != dim:
        return (f"index was built with {meta['model']!r} ({meta['dim']} dims), configured "
                f"{settings.embed_model!r}: {REINDEX_HINT}")
    if "doc_prefix" in meta and meta["doc_prefix"] != settings.embed_doc_prefix:
        return (f"document prefix changed ({meta['doc_prefix']!r} -> {settings.embed_doc_prefix!r}): "
                f"{REINDEX_HINT}")
    return None
