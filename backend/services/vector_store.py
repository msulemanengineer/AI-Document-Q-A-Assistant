"""
STEP 5 + 8 of the RAG pipeline: store vectors, then search them.

FAISS (Facebook AI Similarity Search) is a C++ library that finds the
nearest vectors to a query vector very quickly.

INDEX CHOICE: `IndexFlatIP`
  * "Flat"  = store every vector and compare against all of them (exact
              search, no approximation). Perfect recall; fine for the
              thousands-of-chunks scale a single PDF produces.
  * "IP"    = Inner Product. Because `embedding_service` normalises every
              vector to length 1, the inner product IS the cosine similarity,
              so scores land in a readable -1..1 range where 1 = identical.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from backend.config import settings
from backend.services.chunking_service import Chunk
from backend.utils.errors import VectorStoreError


@dataclass
class SearchResult:
    """One chunk returned by a similarity search, with its score."""

    chunk_id: int
    text: str
    page: int
    score: float  # cosine similarity, 1.0 = identical meaning


class VectorStore:
    """An in-memory FAISS index plus the chunks it was built from.

    FAISS only stores vectors and returns their positions. It knows nothing
    about text, so we keep `self.chunks` alongside and map a returned
    position back to the original chunk.
    """

    def __init__(self) -> None:
        self.index = None  # faiss.IndexFlatIP, created in build()
        self.chunks: List[Chunk] = []
        self.dimension: Optional[int] = None

    # ---------- build ----------

    def build(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """Create the FAISS index from chunks and their embeddings."""
        if not chunks:
            raise VectorStoreError("Cannot build an index with no chunks.")
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise VectorStoreError(
                f"Expected one vector per chunk, got {vectors.shape} "
                f"for {len(chunks)} chunks."
            )

        try:
            import faiss

            vectors = np.ascontiguousarray(vectors, dtype="float32")
            self.dimension = int(vectors.shape[1])
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(vectors)
            self.chunks = list(chunks)
        except Exception as exc:
            self.clear()
            raise VectorStoreError(f"Failed to build the FAISS index: {exc}") from exc

    # ---------- search ----------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[SearchResult]:
        """Return the `top_k` most similar chunks to the query vector.

        Args:
            query_vector:   shape (1, dim), already normalised.
            top_k:          how many chunks to retrieve. Too small risks
                            missing the answer; too large floods the prompt
                            with noise and costs more tokens.
            min_similarity: drop results scoring below this. This is what
                            lets the system answer "not found in the document"
                            instead of forcing a bad match.
        """
        if not self.is_ready():
            raise VectorStoreError("The vector index is empty. Upload a document first.")

        # `is None` rather than `or`, so an explicit 0 is rejected below.
        top_k = settings.TOP_K if top_k is None else top_k
        min_similarity = (
            settings.MIN_SIMILARITY if min_similarity is None else min_similarity
        )
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than 0.")

        query_vector = np.ascontiguousarray(query_vector, dtype="float32")
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Question vector has {query_vector.shape[1]} dimensions but the "
                f"index expects {self.dimension}. The embedding model changed; "
                f"please re-upload the document."
            )

        # Never ask FAISS for more neighbours than it holds.
        k = min(top_k, self.index.ntotal)

        try:
            scores, indices = self.index.search(query_vector, k)
        except Exception as exc:
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS uses -1 to pad when fewer than k results exist
                continue
            if float(score) < min_similarity:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    page=chunk.page,
                    score=round(float(score), 4),
                )
            )
        return results

    # ---------- state ----------

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def size(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def clear(self) -> None:
        self.index = None
        self.chunks = []
        self.dimension = None
