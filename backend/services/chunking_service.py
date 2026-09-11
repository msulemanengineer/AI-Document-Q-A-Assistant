"""
STEP 3 of the RAG pipeline: clean text -> overlapping chunks.

WHY CHUNK AT ALL?
  * An embedding model turns any input into ONE fixed-size vector. If you
    embed 50 pages into one vector, every topic is averaged together and the
    vector means nothing specific -- search quality collapses.
  * Embedding models have a hard input limit (all-MiniLM-L6-v2 truncates at
    256 word pieces). Anything past the limit is silently thrown away.
  * We only send the retrieved chunks to the LLM, so smaller chunks mean a
    tighter, cheaper, less distracting prompt.

WHY OVERLAP?
  A fixed cut can land in the middle of a sentence or split a question from
  its answer. Repeating the last `chunk_overlap` characters at the start of
  the next chunk means an idea that straddles a boundary still appears whole
  in at least one chunk.
"""

from dataclasses import dataclass
from typing import List, Tuple

from backend.config import settings


@dataclass
class Chunk:
    """One retrievable piece of the document."""

    chunk_id: int  # position in the document, 0-based
    text: str
    page: int  # 1-based source page, used for citations


def _split_page(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split one page into overlapping windows, preferring sentence ends.

    The window is `chunk_size` characters wide and slides forward by
    `chunk_size - chunk_overlap` characters each step. Before cutting, we look
    backwards a little for a sentence boundary so chunks end on a full thought.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap  # guaranteed > 0 by chunk_document()

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Not the final chunk? Try to end on a sentence boundary instead of
        # mid-word. We only search the last 20% of the window so a chunk is
        # never made drastically shorter than requested.
        if end < len(text):
            window_start = max(start, end - chunk_size // 5)
            boundary = max(
                text.rfind(". ", window_start, end),
                text.rfind("! ", window_start, end),
                text.rfind("? ", window_start, end),
                text.rfind("\n", window_start, end),
            )
            if boundary > start:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break
        # Step forward from the ACTUAL end so the overlap is honoured even
        # when the sentence-boundary search moved `end` backwards.
        start = max(start + step, end - chunk_overlap)

    return chunks


def chunk_document(
    pages: List[Tuple[int, str]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Chunk]:
    """Turn `(page_number, text)` pairs into a flat list of `Chunk` objects.

    Chunking page-by-page (rather than over one giant string) keeps the page
    number attached to every chunk, which is what makes source citation possible.

    Args:
        pages:         output of `pdf_service.extract_pages`.
        chunk_size:    characters per chunk. Bigger = more context per chunk
                       but less precise retrieval. Defaults to settings.
        chunk_overlap: characters repeated between neighbouring chunks.
                       Must be smaller than chunk_size.
    """
    # `is None` rather than `or`, so an explicit 0 is rejected below instead of
    # being silently replaced by the default.
    chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
    chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        # Otherwise the window would never move forward -> infinite loop.
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: List[Chunk] = []
    for page_number, page_text in pages:
        for piece in _split_page(page_text, chunk_size, chunk_overlap):
            chunks.append(Chunk(chunk_id=len(chunks), text=piece, page=page_number))

    return chunks
