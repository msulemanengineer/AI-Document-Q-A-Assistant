"""Tests for PDF text extraction and text cleaning."""

import pytest

from backend.services import pdf_service
from backend.utils.errors import EmptyPDFError, InvalidPDFError
from backend.utils.text_cleaning import clean_text, is_meaningful


def test_extract_pages_returns_one_entry_per_page(sample_pdf_bytes, page_texts):
    pages = pdf_service.extract_pages(sample_pdf_bytes)

    assert len(pages) == len(page_texts)
    # Page numbers must be 1-based and in order -- citations depend on this.
    assert [number for number, _ in pages] == [1, 2, 3]


def test_extract_pages_recovers_the_original_text(sample_pdf_bytes):
    pages = dict(pdf_service.extract_pages(sample_pdf_bytes))

    assert "Eiffel Tower" in pages[1]
    assert "Photosynthesis" in pages[2]
    assert "4.2 million" in pages[3]


def test_extract_text_joins_all_pages(sample_pdf_bytes):
    text = pdf_service.extract_text(sample_pdf_bytes)

    assert "Eiffel Tower" in text
    assert "Photosynthesis" in text


def test_count_pages(sample_pdf_bytes):
    assert pdf_service.count_pages(sample_pdf_bytes) == 3


def test_empty_bytes_raise_invalid_pdf():
    with pytest.raises(InvalidPDFError):
        pdf_service.extract_pages(b"")


def test_non_pdf_bytes_raise_invalid_pdf():
    with pytest.raises(InvalidPDFError):
        pdf_service.extract_pages(b"this is a text file, not a PDF")


def test_pdf_without_text_raises_empty_pdf(blank_pdf_bytes):
    # This is the "scanned document" case: the PDF is valid but has no text.
    with pytest.raises(EmptyPDFError):
        pdf_service.extract_pages(blank_pdf_bytes)


# ---------------- text cleaning ----------------


def test_clean_text_rejoins_hyphenated_line_breaks():
    assert clean_text("inter-\nnational") == "international"


def test_clean_text_unwraps_single_newlines_but_keeps_paragraphs():
    cleaned = clean_text("line one\nline two\n\nnew paragraph")

    assert "line one line two" in cleaned
    assert "\n\n" in cleaned


def test_clean_text_collapses_extra_whitespace():
    assert clean_text("too    many\t\tspaces") == "too many spaces"


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text("   \n  ") == ""


def test_is_meaningful_rejects_near_empty_text():
    assert is_meaningful("This sentence is long enough to index.")
    assert not is_meaningful("7")
    assert not is_meaningful("   ")
