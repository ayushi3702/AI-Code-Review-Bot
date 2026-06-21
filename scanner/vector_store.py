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
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._embeddings = get_embeddings()
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ── Indexing ─────────────────────────────────────────────────────────────
    def index(self, chunks: list[Chunk]) -> int:
        """Embed and store chunks. Returns number indexed."""
        if not chunks:
            return 0

        total = 0
        for i in range(0, len(chunks), _EMBED_BATCH):
            batch = chunks[i:i + _EMBED_BATCH]
            texts = [c.content for c in batch]
            vectors = self._embeddings.embed_documents(texts)
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
            logger.info("Indexed %d/%d chunks", total, len(chunks))
        return total

    # ── Retrieval ────────────────────────────────────────────────────────────
    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or RETRIEVAL_TOP_K
        vector = self._embeddings.embed_query(text)
        res = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_chunks(res)

    def query_many(self, texts: list[str], top_k: int | None = None) -> list[RetrievedChunk]:
        """Run several concern phrases and merge, de-duplicating by chunk identity."""
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
        vector = self._embeddings.embed_query(content)
        res = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_chunks(res)

    def count(self) -> int:
        return self._collection.count()

    def delete(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception as e:
            logger.warning("Failed to delete collection %s: %s", self.collection_name, e)

    @staticmethod
    def _to_chunks(res) -> list[RetrievedChunk]:
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
