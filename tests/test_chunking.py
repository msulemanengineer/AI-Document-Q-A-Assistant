"""Tests for the chunking service."""

import pytest

from backend.services.chunking_service import Chunk, chunk_document

LONG_PAGE = (
    "Retrieval augmented generation combines search with language models. " * 40
)


def test_short_page_becomes_a_single_chunk():
    chunks = chunk_document([(1, "A short sentence.")], chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].text == "A short sentence."


def test_long_page_is_split_into_several_chunks():
    chunks = chunk_document([(1, LONG_PAGE)], chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    # No chunk should be wildly bigger than requested.
    assert all(len(c.text) <= 300 for c in chunks)


def test_chunk_ids_are_sequential_and_pages_are_preserved():
    pages = [(1, LONG_PAGE), (2, LONG_PAGE), (7, "Short page seven text here.")]
    chunks = chunk_document(pages, chunk_size=300, chunk_overlap=50)

    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    # The page number must survive chunking -- it is what powers citations.
    assert {c.page for c in chunks} == {1, 2, 7}
    assert chunks[-1].page == 7


def test_neighbouring_chunks_actually_overlap():
    # Distinct words so we can prove text is repeated across the boundary.
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_document([(1, text)], chunk_size=400, chunk_overlap=100)

    assert len(chunks) > 1
    tail = chunks[0].text[-60:]
    # Some part of the end of chunk 0 must reappear inside chunk 1.
    assert any(word in chunks[1].text for word in tail.split())


def test_no_chunk_is_empty_or_whitespace():
    chunks = chunk_document([(1, LONG_PAGE)], chunk_size=200, chunk_overlap=40)

    assert all(c.text.strip() for c in chunks)


def test_chunking_covers_the_whole_document():
    text = " ".join(f"token{i}" for i in range(300))
    chunks = chunk_document([(1, text)], chunk_size=350, chunk_overlap=50)

    joined = " ".join(c.text for c in chunks)
    assert "token0" in joined
    assert "token299" in joined


def test_overlap_must_be_smaller_than_chunk_size():
    # Otherwise the sliding window would never advance -> infinite loop.
    with pytest.raises(ValueError):
        chunk_document([(1, LONG_PAGE)], chunk_size=100, chunk_overlap=100)


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_document([(1, LONG_PAGE)], chunk_size=0, chunk_overlap=0)


def test_empty_pages_produce_no_chunks():
    assert chunk_document([]) == []
    assert chunk_document([(1, "   ")]) == []


def test_returns_chunk_objects():
    chunks = chunk_document([(1, "Some text for the chunker to handle.")])

    assert all(isinstance(c, Chunk) for c in chunks)
