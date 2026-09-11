"""
Tests for embedding generation and FAISS vector search.

These tests load the real all-MiniLM-L6-v2 model, so the FIRST run needs
internet to download it (~80 MB). After that it is cached and runs offline.
They are the tests that prove semantic search actually works: the query
"Where is the Eiffel Tower?" must retrieve the Paris chunk even though it
shares almost no keywords with it.
"""

import numpy as np
import pytest

from backend.services.chunking_service import Chunk
from backend.services.embedding_service import embed_query, embed_texts, get_dimension
from backend.services.vector_store import VectorStore
from backend.utils.errors import EmbeddingError, VectorStoreError

CHUNKS = [
    Chunk(0, "The Eiffel Tower is located in Paris, France and was built in 1889.", 1),
    Chunk(1, "Photosynthesis lets plants turn sunlight into chemical energy.", 2),
    Chunk(2, "The company reported total revenue of 4.2 million dollars in 2023.", 3),
]


@pytest.fixture(scope="module")
def store() -> VectorStore:
    """A FAISS index built from the three test chunks."""
    vectors = embed_texts([c.text for c in CHUNKS])
    vector_store = VectorStore()
    vector_store.build(CHUNKS, vectors)
    return vector_store


# ---------------- embeddings ----------------


def test_embeddings_have_the_expected_shape_and_type():
    vectors = embed_texts(["first text", "second text"])

    assert vectors.shape == (2, get_dimension())
    assert vectors.dtype == np.float32  # FAISS requires float32


def test_all_minilm_produces_384_dimensions():
    assert get_dimension() == 384


def test_embeddings_are_normalised_to_unit_length():
    # This is what makes FAISS inner-product search equal cosine similarity.
    vectors = embed_texts(["some sentence", "another sentence"])
    norms = np.linalg.norm(vectors, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_similar_sentences_score_higher_than_unrelated_ones():
    vectors = embed_texts(
        [
            "A dog is running in the park.",
            "A puppy is playing outside on the grass.",
            "Quarterly financial results were published today.",
        ]
    )
    similar = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])

    assert similar > unrelated


def test_empty_input_raises_embedding_error():
    with pytest.raises(EmbeddingError):
        embed_texts([])
    with pytest.raises(EmbeddingError):
        embed_texts(["valid text", "   "])
    with pytest.raises(EmbeddingError):
        embed_query("")


def test_embed_query_returns_a_single_row():
    assert embed_query("What is this about?").shape == (1, get_dimension())


# ---------------- vector store ----------------


def test_store_is_ready_after_building(store):
    assert store.is_ready()
    assert store.size() == len(CHUNKS)
    assert store.dimension == 384


def test_semantic_search_finds_the_right_chunk_without_shared_keywords(store):
    # "Where is the famous iron tower?" shares no distinctive words with the
    # stored sentence, so a keyword search would fail here. Embeddings do not.
    results = store.search(embed_query("Where is the famous iron tower?"), top_k=1)

    assert len(results) == 1
    assert "Eiffel" in results[0].text
    assert results[0].page == 1


def test_search_on_a_different_topic_finds_a_different_chunk(store):
    results = store.search(embed_query("How much money did the business make?"), top_k=1)

    assert "revenue" in results[0].text
    assert results[0].page == 3


def test_results_are_sorted_by_descending_similarity(store):
    results = store.search(embed_query("plants and sunlight"), top_k=3, min_similarity=0.0)
    scores = [r.score for r in results]

    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_the_number_of_results(store):
    assert len(store.search(embed_query("energy"), top_k=2, min_similarity=0.0)) == 2


def test_top_k_larger_than_the_index_is_safe(store):
    results = store.search(embed_query("energy"), top_k=50, min_similarity=0.0)

    assert len(results) == len(CHUNKS)


def test_high_similarity_threshold_filters_out_irrelevant_chunks(store):
    # Nothing in the index is about this, so a strict threshold returns nothing.
    # This is the mechanism behind "not found in the document".
    results = store.search(
        embed_query("the rules of professional ice hockey"), min_similarity=0.75
    )

    assert results == []


def test_scores_are_cosine_similarities_in_range(store):
    results = store.search(embed_query("Paris"), top_k=3, min_similarity=0.0)

    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_searching_an_empty_store_raises():
    with pytest.raises(VectorStoreError):
        VectorStore().search(embed_query("anything"))


def test_building_with_no_chunks_raises():
    with pytest.raises(VectorStoreError):
        VectorStore().build([], np.zeros((0, 384), dtype="float32"))


def test_mismatched_vector_count_raises():
    with pytest.raises(VectorStoreError):
        VectorStore().build(CHUNKS, np.zeros((2, 384), dtype="float32"))


def test_wrong_query_dimension_raises(store):
    with pytest.raises(VectorStoreError):
        store.search(np.zeros((1, 128), dtype="float32"))


def test_clear_empties_the_store():
    vectors = embed_texts([c.text for c in CHUNKS])
    vector_store = VectorStore()
    vector_store.build(CHUNKS, vectors)
    vector_store.clear()

    assert not vector_store.is_ready()
    assert vector_store.size() == 0
