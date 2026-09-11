"""
Shared pytest fixtures.

The most important one is `sample_pdf_bytes`: it builds a small PDF in memory
so the tests never depend on a file being present on disk.
"""

import io
import sys
from pathlib import Path

import pytest

# Make `import backend...` work when pytest is run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_pdf(pages: list[str]) -> bytes:
    """Build a minimal, valid PDF whose pages contain the given lines of text.

    Written by hand rather than with a PDF library so the tests do not need an
    extra dependency. It only uses the handful of PDF objects pypdf needs to
    extract text: a catalog, a pages tree, page objects, a font, and content
    streams using the Tj text operator.
    """
    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)  # object numbers are 1-based

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    content_ids: list[int] = []
    for text in pages:
        # Escape the characters that are special inside a PDF string literal.
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode("latin-1")
        content_ids.append(
            add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        )

    pages_tree_id = len(objects) + len(pages) + 1
    for content_id in content_ids:
        page_ids.append(
            add(
                f"<< /Type /Page /Parent {pages_tree_id} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>".encode()
            )
        )

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_tree_id} 0 R >>".encode())

    # Assemble the file body, recording each object's byte offset for the xref.
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")

    xref_position = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n".encode()
    )
    return out.getvalue()


# Text used by the tests. Each page holds one clearly distinct topic so we can
# assert that semantic search retrieves the RIGHT page, not just any page.
PAGE_TEXTS = [
    "The Eiffel Tower is located in Paris, France and was completed in 1889.",
    "Photosynthesis is the process plants use to convert sunlight into energy.",
    "The company reported total revenue of 4.2 million dollars in the year 2023.",
]


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """A valid 3-page PDF with known, searchable text."""
    return _make_pdf(PAGE_TEXTS)


@pytest.fixture(scope="session")
def page_texts() -> list[str]:
    return PAGE_TEXTS


@pytest.fixture(scope="session")
def blank_pdf_bytes() -> bytes:
    """A structurally valid PDF whose pages contain no extractable text."""
    return _make_pdf(["", " "])
