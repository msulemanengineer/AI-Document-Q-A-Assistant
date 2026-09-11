"""
STEP 4 of the RAG pipeline: text -> vectors (embeddings).

An embedding is a list of numbers that represents the MEANING of a piece of
text. Texts that mean similar things get vectors that point in similar
directions, which is what lets us search by meaning instead of by keyword.

Model: all-MiniLM-L6-v2
  * 384 dimensions, ~80 MB, runs comfortably on CPU.
  * Downloaded automatically from Hugging Face on first use and cached in
    ~/.cache/huggingface, so only the first run needs internet.
"""

from typing import List, Optional

import numpy as np

from backend.config import settings
from backend.utils.errors import EmbeddingError

# The model is loaded lazily and kept in this module-level variable so the
# ~80 MB of weights are loaded once per process, not once per request.
_model = None
_model_name: Optional[str] = None


def get_model(model_name: Optional[str] = None):
    """Load (once) and return the sentence-transformers model."""
    global _model, _model_name
    model_name = model_name or settings.EMBEDDING_MODEL

    if _model is not None and _model_name == model_name:
        return _model

    try:
        # Imported here, not at module top level, because importing
        # sentence_transformers pulls in torch and takes a few seconds.
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
        _model_name = model_name
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load the embedding model '{model_name}'. "
            f"The first run needs internet access to download it. Details: {exc}"
        ) from exc

    return _model


def get_dimension() -> int:
    """Vector length produced by the current model (384 for all-MiniLM-L6-v2)."""
    return int(get_model().get_sentence_embedding_dimension())


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts into a float32 array of shape (n_texts, dim).

    `normalize_embeddings=True` scales every vector to length 1. That is the
    key trick that lets FAISS's fast inner-product search return exactly the
    COSINE SIMILARITY, because for unit vectors  dot(a, b) == cos(a, b).
    """
    if not texts:
        raise EmbeddingError("No text was given to the embedding model.")
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise EmbeddingError("Cannot embed empty text.")

    model = get_model()
    try:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

    # FAISS requires contiguous float32 input.
    return np.asarray(vectors, dtype="float32")


def embed_query(question: str) -> np.ndarray:
    """Embed a single question into shape (1, dim), ready for FAISS search.

    The question goes through the SAME model as the chunks -- that is
    essential, because two different models produce vectors that live in two
    different, uncomparable spaces.
    """
    if not question or not question.strip():
        raise EmbeddingError("Cannot embed an empty question.")
    return embed_texts([question.strip()])
