"""Vector + keyword stores.

`InMemoryStore` backs tests and the local demo. `PgVectorStore` is the
production store: one Postgres holds the vectors (pgvector, HNSW index), the
full-text index (generated tsvector, GIN) and the metadata (JSONB, GIN), so
hybrid retrieval with metadata filters is one round trip and one system to
back up, instead of a separate vector database to keep in sync.

Ingestion is idempotent: each document stores the hash of its cleaned text,
and re-ingesting an unchanged document is a no-op.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import numpy as np

from .bm25 import BM25Index
from .documents import Chunk

LIST_FIELDS = {"skills"}
Filters = dict[str, Any]


def matches(meta: dict[str, Any], filters: Filters | None) -> bool:
    if not filters:
        return True
    for key, want in filters.items():
        have = meta.get(key)
        if isinstance(have, list):
            wanted = want if isinstance(want, list) else [want]
            if not set(wanted) <= set(have):
                return False
        elif have != want:
            return False
    return True


class VectorStore(Protocol):
    def document_hash(self, doc_id: str) -> str | None: ...

    def upsert(self, doc_id: str, content_hash: str, doc_meta: dict[str, Any],
               chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def delete_document(self, doc_id: str) -> None: ...

    def vector_search(self, qvec: np.ndarray, k: int,
                      filters: Filters | None = None) -> list[tuple[Chunk, float]]: ...

    def keyword_search(self, query: str, k: int,
                       filters: Filters | None = None) -> list[tuple[Chunk, float]]: ...

    def documents(self, filters: Filters | None = None) -> list[tuple[str, dict[str, Any]]]: ...

    def stats(self) -> dict[str, int]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vecs: dict[str, np.ndarray] = {}
        self._docs: dict[str, tuple[str, dict[str, Any], list[str]]] = {}
        self._bm25 = BM25Index()

    def document_hash(self, doc_id: str) -> str | None:
        entry = self._docs.get(doc_id)
        return entry[0] if entry else None

    def upsert(self, doc_id, content_hash, doc_meta, chunks, vectors) -> None:
        self.delete_document(doc_id)
        ids = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._vecs[chunk.chunk_id] = vec
            self._bm25.add(chunk.chunk_id, chunk.text)
            ids.append(chunk.chunk_id)
        self._docs[doc_id] = (content_hash, doc_meta, ids)

    def delete_document(self, doc_id: str) -> None:
        entry = self._docs.pop(doc_id, None)
        if not entry:
            return
        for cid in entry[2]:
            self._chunks.pop(cid, None)
            self._vecs.pop(cid, None)
            self._bm25.remove(cid)

    def _eligible(self, filters: Filters | None) -> list[str]:
        return [cid for cid, c in self._chunks.items() if matches(c.metadata, filters)]

    def vector_search(self, qvec, k, filters=None):
        ids = self._eligible(filters)
        if not ids:
            return []
        mat = np.stack([self._vecs[i] for i in ids])
        sims = mat @ qvec
        order = np.argsort(-sims)[:k]
        return [(self._chunks[ids[i]], float(sims[i])) for i in order]

    def keyword_search(self, query, k, filters=None):
        keys = set(self._eligible(filters))
        return [(self._chunks[cid], s) for cid, s in self._bm25.search(query, keys, k)]

    def documents(self, filters=None):
        return [(d, meta) for d, (_, meta, _) in self._docs.items() if matches(meta, filters)]

    def stats(self) -> dict[str, int]:
        return {"documents": len(self._docs), "chunks": len(self._chunks)}


_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS bw_documents (
    doc_id        text PRIMARY KEY,
    content_hash  text NOT NULL,
    metadata      jsonb NOT NULL DEFAULT '{}',
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS bw_chunks (
    chunk_id   text PRIMARY KEY,
    doc_id     text NOT NULL REFERENCES bw_documents(doc_id) ON DELETE CASCADE,
    position   int  NOT NULL,
    section    text NOT NULL DEFAULT '',
    text       text NOT NULL,
    metadata   jsonb NOT NULL DEFAULT '{}',
    embedding  vector(%(dim)s) NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS bw_chunks_embedding_hnsw
    ON bw_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS bw_chunks_tsv_gin ON bw_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS bw_chunks_meta_gin ON bw_chunks USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS bw_chunks_doc ON bw_chunks (doc_id);
"""


class PgVectorStore:  # pragma: no cover - exercised by the integration job against a real DB
    def __init__(self, dsn: str, dim: int = 384, pool_size: int = 5) -> None:
        from pgvector.psycopg import register_vector
        from psycopg_pool import ConnectionPool

        self.dim = dim
        self.pool = ConnectionPool(dsn, min_size=1, max_size=pool_size,
                                   configure=register_vector, open=True)
        with self.pool.connection() as conn:
            conn.execute(_SCHEMA % {"dim": int(dim)})

    @staticmethod
    def _filter_sql(filters: Filters | None) -> tuple[str, list[Any]]:
        if not filters:
            return "", []
        doc = {k: ([v] if k in LIST_FIELDS and not isinstance(v, list) else v)
               for k, v in filters.items()}
        return " AND c.metadata @> %s::jsonb", [json.dumps(doc)]

    @staticmethod
    def _row(r: tuple) -> Chunk:
        return Chunk(chunk_id=r[0], doc_id=r[1], position=r[2], section=r[3],
                     text=r[4], metadata=r[5])

    def document_hash(self, doc_id: str) -> str | None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT content_hash FROM bw_documents WHERE doc_id = %s",
                               (doc_id,)).fetchone()
        return row[0] if row else None

    def upsert(self, doc_id, content_hash, doc_meta, chunks, vectors) -> None:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO bw_documents (doc_id, content_hash, metadata) VALUES (%s, %s, %s) "
                "ON CONFLICT (doc_id) DO UPDATE SET content_hash = EXCLUDED.content_hash, "
                "metadata = EXCLUDED.metadata, updated_at = now()",
                (doc_id, content_hash, json.dumps(doc_meta)))
            conn.execute("DELETE FROM bw_chunks WHERE doc_id = %s", (doc_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO bw_chunks (chunk_id, doc_id, position, section, text, metadata, "
                    "embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [(c.chunk_id, c.doc_id, c.position, c.section, c.text,
                      json.dumps(c.metadata), v) for c, v in zip(chunks, vectors, strict=True)])

    def delete_document(self, doc_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM bw_documents WHERE doc_id = %s", (doc_id,))

    def vector_search(self, qvec, k, filters=None):
        where, params = self._filter_sql(filters)
        sql = ("SELECT c.chunk_id, c.doc_id, c.position, c.section, c.text, c.metadata, "
               "1 - (c.embedding <=> %s) AS sim FROM bw_chunks c WHERE true" + where +
               " ORDER BY c.embedding <=> %s LIMIT %s")
        with self.pool.connection() as conn:
            conn.execute("SET LOCAL hnsw.ef_search = 80")
            rows = conn.execute(sql, [qvec, *params, qvec, k]).fetchall()
        return [(self._row(r), float(r[6])) for r in rows]

    def keyword_search(self, query, k, filters=None):
        where, params = self._filter_sql(filters)
        sql = ("SELECT c.chunk_id, c.doc_id, c.position, c.section, c.text, c.metadata, "
               "ts_rank_cd(c.tsv, q) AS rank FROM bw_chunks c, "
               "websearch_to_tsquery('english', %s) q WHERE c.tsv @@ q" + where +
               " ORDER BY rank DESC LIMIT %s")
        with self.pool.connection() as conn:
            rows = conn.execute(sql, [query, *params, k]).fetchall()
        return [(self._row(r), float(r[6])) for r in rows]

    def documents(self, filters=None):
        sql = "SELECT doc_id, metadata FROM bw_documents"
        params: list[Any] = []
        if filters:
            doc = {k: ([v] if k in LIST_FIELDS and not isinstance(v, list) else v)
                   for k, v in filters.items()}
            sql += " WHERE metadata @> %s::jsonb"
            params.append(json.dumps(doc))
        with self.pool.connection() as conn:
            return [(r[0], r[1]) for r in conn.execute(sql, params).fetchall()]

    def stats(self) -> dict[str, int]:
        with self.pool.connection() as conn:
            d = conn.execute("SELECT count(*) FROM bw_documents").fetchone()[0]
            c = conn.execute("SELECT count(*) FROM bw_chunks").fetchone()[0]
        return {"documents": d, "chunks": c}
