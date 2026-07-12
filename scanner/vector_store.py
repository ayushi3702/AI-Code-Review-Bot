"""Vector store wrapper around ChromaDB.

Each scan gets its own persistent Chroma collection named after the scan_id.
We embed chunks with Azure OpenAI embeddings and store the chunk text plus
metadata (file path, symbol, line range, language). Agents never load the whole
repo — they call `query()` with concern-specific phrases and get back only the
most relevant chunks, so cost stays flat as the repo grows.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

import chromadb

from core.config import CHROMA_PERSIST_DIR, get_embeddings, RETRIEVAL_TOP_K
from scanner.chunker import Chunk

logger = logging.getLogger(__name__)

_EMBED_BATCH = 64


@dataclass
class RetrievedChunk:
    file_path: str
    symbol: str | None
    language: str
    start_line: int
    end_line: int
    content: str
    distance: float


class VectorStore:
    """Per-scan ChromaDB vector store for semantic code retrieval.

    Each scan gets its own persistent Chroma collection named after the scan ID
    (e.g. ``scan_abc123``).  Chunks are embedded with the configured embeddings
    provider and stored with metadata (file path, symbol, line range, language).

    Agents query the store with natural-language concern phrases and receive
    back the most semantically similar code chunks, so prompt token cost stays
    flat as the repository grows regardless of its total size.
    """
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._embeddings = get_embeddings()
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ── Indexing ─────────────────────────────────────────────────────────────
    def index(self, chunks: list[Chunk]) -> int:
        """Embed and store chunks in the collection in batches of :data:`_EMBED_BATCH`.

        Processes chunks in batches to stay within the embedding API’s per-request
        limit and to provide progress logging.  Each batch is embedded atomically
        — a failure in one batch raises immediately without partial writes.

        Args:
            chunks: List of :class:`~scanner.chunker.Chunk` objects to embed.

        Returns:
            Total number of chunks successfully indexed.

        Raises:
            Exception: Re-raises any embedding or ChromaDB error after logging.
        """
        if not chunks:
            return 0

        total = 0
        for i in range(0, len(chunks), _EMBED_BATCH):
            batch = chunks[i:i + _EMBED_BATCH]
            texts = [c.content for c in batch]
            try:
                vectors = self._embeddings.embed_documents(texts)
            except Exception as e:
                logger.error(
                    "Failed to embed batch %d–%d for collection %s: %s",
                    i, i + len(batch), self.collection_name, e, exc_info=True,
                )
                raise
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=vectors,
                metadatas=[{
                    "file_path": c.file_path,
                    "symbol": c.symbol or "",
                    "language": c.language,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                } for c in batch],
            )
            total += len(batch)
            logger.info(
                "Indexed %d/%d chunks into collection %s",
                total, len(chunks), self.collection_name,
            )
        return total

    # ── Retrieval ────────────────────────────────────────────────────────────
    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve the top-k most semantically similar chunks for a query phrase.

        Embeds the query text and performs an approximate nearest-neighbour
        search using ChromaDB's HNSW index (cosine distance space).

        Args:
            text:  Natural-language query phrase (e.g. ``'SQL query raw string'``).
            top_k: Number of results to return; defaults to
                   :data:`~core.config.RETRIEVAL_TOP_K`.

        Returns:
            List of :class:`RetrievedChunk` objects ordered by ascending cosine
            distance (most similar first).
        """
        top_k = top_k or RETRIEVAL_TOP_K
        vector = self._embeddings.embed_query(text)
        res = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_chunks(res)

    def query_many(self, texts: list[str], top_k: int | None = None) -> list[RetrievedChunk]:
        """Run several concern phrases and merge the results, de-duplicating by chunk identity.

        Each phrase is queried independently and the results are merged in order.
        Duplicate chunks (same file path and start line) are removed so the
        prompt does not contain repeated context.

        Args:
            texts: List of query phrases; each is embedded and queried separately.
            top_k: Per-phrase result limit; defaults to
                   :data:`~core.config.RETRIEVAL_TOP_K`.

        Returns:
            Merged, de-duplicated list of :class:`RetrievedChunk` objects.
        """
        seen: set[tuple] = set()
        merged: list[RetrievedChunk] = []
        for t in texts:
            for rc in self.query(t, top_k=top_k):
                key = (rc.file_path, rc.start_line)
                if key not in seen:
                    seen.add(key)
                    merged.append(rc)
        return merged

    def similar_to(self, content: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Find chunks whose embedding is nearest to the given content string.

        Used by the architecture agent to detect near-duplicate code across
        files.  Embedding the content directly (rather than a query phrase)
        finds chunks that are semantically near-identical to the probe.

        Args:
            content: Raw code content to use as the query vector.
            top_k:   Number of nearest neighbours to return.

        Returns:
            List of :class:`RetrievedChunk` objects ordered by ascending cosine
            distance.
        """
        vector = self._embeddings.embed_query(content)
        res = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_chunks(res)

    def count(self) -> int:
        """Return the total number of chunks currently stored in this collection."""
        return self._collection.count()

    def delete(self) -> None:
        """Delete the entire ChromaDB collection for this scan.

        Called during cleanup to free disk space once a scan’s results have
        been committed to the relational database.  Errors are logged as
        warnings rather than raised, since a missing collection is harmless.
        """
        try:
            self._client.delete_collection(self.collection_name)
        except Exception as e:
            logger.warning("Failed to delete collection %s: %s", self.collection_name, e)

    @staticmethod
    def _to_chunks(res) -> list[RetrievedChunk]:
        """Convert a raw ChromaDB query result dict into :class:`RetrievedChunk` objects.

        ChromaDB returns a dict with parallel lists under ``'ids'``,
        ``'documents'``, ``'metadatas'``, and ``'distances'`` keys.  This
        helper zips those lists together into typed dataclass instances.

        Args:
            res: Raw result dict from ``collection.query()``.

        Returns:
            List of :class:`RetrievedChunk` objects, or ``[]`` if the result is empty.
        """
        out: list[RetrievedChunk] = []
        if not res.get("ids") or not res["ids"][0]:
            return out
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[0] * len(docs)])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(RetrievedChunk(
                file_path=meta.get("file_path", ""),
                symbol=meta.get("symbol") or None,
                language=meta.get("language", ""),
                start_line=int(meta.get("start_line", 0)),
                end_line=int(meta.get("end_line", 0)),
                content=doc,
                distance=float(dist),
            ))
        return out
