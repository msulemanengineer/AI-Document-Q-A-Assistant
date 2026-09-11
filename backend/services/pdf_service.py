"""
STEP 1 + 2 of the RAG pipeline: PDF -> raw text -> clean text.

Uses `pypdf`, a pure-Python library, so there is nothing to compile and the
project installs the same way on Windows, macOS and Linux.
"""

import io
from typing import List, Tuple

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from backend.utils.errors import EmptyPDFError, InvalidPDFError
from backend.utils.text_cleaning import clean_text, is_meaningful


def extract_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract cleaned text from a PDF, one entry per page.

    Returns a list of `(page_number, text)` where page numbers start at 1.
    Keeping text per page is what lets us later tell the user
    "this answer came from page 7".

    Raises:
        InvalidPDFError: the bytes are not a PDF we can open.
        EmptyPDFError:   the PDF opened but has no extractable text.
    """
    if not pdf_bytes:
        raise InvalidPDFError("The uploaded file is empty.")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except (PdfReadError, OSError, ValueError) as exc:
        raise InvalidPDFError(f"Could not read the PDF file: {exc}") from exc

    # An encrypted PDF may open but return empty pages; try the common
    # "empty password" case before giving up.
    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise InvalidPDFError(
                    "This PDF is password protected. Please upload an unlocked PDF."
                )
        except NotImplementedError as exc:
            raise InvalidPDFError(
                "This PDF uses an unsupported encryption method."
            ) from exc

    if len(reader.pages) == 0:
        raise EmptyPDFError("The PDF has no pages.")

    pages: List[Tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            # One broken page should not kill the whole upload.
            raw = ""
        cleaned = clean_text(raw)
        if is_meaningful(cleaned):
            pages.append((page_number, cleaned))

    if not pages:
        # Almost always a scanned/image-only PDF. It needs OCR, which this
        # project does not do -- so say that plainly instead of failing silently.
        raise EmptyPDFError(
            "No readable text was found in this PDF. "
            "It is probably a scanned image; this app cannot run OCR."
        )

    return pages


def extract_text(pdf_bytes: bytes) -> str:
    """Convenience helper: the whole document as one cleaned string."""
    return "\n\n".join(text for _, text in extract_pages(pdf_bytes))


def count_pages(pdf_bytes: bytes) -> int:
    """Number of pages in the PDF (including pages with no text)."""
    try:
        return len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception as exc:
        raise InvalidPDFError(f"Could not read the PDF file: {exc}") from exc
