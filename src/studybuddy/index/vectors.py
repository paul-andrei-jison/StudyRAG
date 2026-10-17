"""LanceDB vector store. Holds only (chunk id, vector, subject key, doc id): all text lives in SQLite.

Behaviour below was checked against the installed lancedb (0.40): cosine `_distance` is 1 - similarity, a
prefilter restricts the candidates before ranking, and a wrong-sized vector raises.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import lancedb
import pyarrow as pa

from ..errors import IndexStoreError

TABLE = "chunks"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class VectorStore:
    def __init__(self, path: Path, dim: int | None = None):
        """Open (or lazily create) the vector table at `path`.

        An existing table fixes the dimension; a conflicting `dim` raises. A new table needs `dim`, or gets
        it from the first `add`.
        """
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(path))
        self._table = None
        if TABLE in self._table_names():
            self._table = self._db.open_table(TABLE)
            existing = self._table.schema.field("vector").type.list_size
            if dim is not None and dim != existing:
                raise IndexStoreError(f"Vector table has {existing} dimensions, asked for {dim}")
            self.dim: int | None = existing
        else:
            self.dim = dim
            if dim is not None:
                self._create(dim)

    def _table_names(self) -> list[str]:
        return list(self._db.list_tables().tables)

    def _create(self, dim: int) -> None:
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("subject", pa.string()),
            pa.field("doc_id", pa.int64()),
        ])
        self._table = self._db.create_table(TABLE, schema=schema, mode="overwrite")
        self.dim = dim

    def add(self, ids: Sequence[int], vectors: Sequence[Sequence[float]], subjects: Sequence[str],
            doc_ids: Sequence[int]) -> None:
        if not (len(ids) == len(vectors) == len(subjects) == len(doc_ids)):
            raise IndexStoreError("ids, vectors, subjects and doc_ids must have the same length")
        if not ids:
            return
        if self._table is None:
            self._create(len(vectors[0]))
        for vec in vectors:
            if len(vec) != self.dim:
                raise IndexStoreError(f"Vector has {len(vec)} dimensions, the index uses {self.dim}")
        self._table.add([{"id": int(i), "vector": [float(x) for x in v], "subject": s, "doc_id": int(d)}
                         for i, v, s, d in zip(ids, vectors, subjects, doc_ids)])

    def delete(self, ids: Sequence[int]) -> None:
        if self._table is not None and ids:
            self._table.delete("id IN (" + ",".join(str(int(i)) for i in ids) + ")")

    def update_subject(self, doc_id: int, subject_key: str) -> None:
        if self._table is not None:
            self._table.update(where=f"doc_id = {int(doc_id)}", values={"subject": subject_key})

    def count(self) -> int:
        return self._table.count_rows() if self._table is not None else 0

    def ids(self) -> set[int]:
        if self._table is None:
            return set()
        return set(self._table.to_arrow().column("id").to_pylist())

    def search(self, vector: Sequence[float], n: int, subject_key: str | None = None) -> list[tuple[int, float]]:
        """Top-n (chunk id, cosine similarity), best first. The subject filter is applied before ranking."""
        if self._table is None or n <= 0:
            return []
        if len(vector) != self.dim:
            raise IndexStoreError(f"Query vector has {len(vector)} dimensions, the index uses {self.dim}")
        query = self._table.search([float(x) for x in vector]).metric("cosine")
        if subject_key:
            query = query.where(f"subject = {_quote(subject_key)}", prefilter=True)
        return [(int(r["id"]), 1.0 - float(r["_distance"])) for r in query.limit(n).to_list()]
