"""
Text cleaning helpers.

Raw PDF text is messy: hyphenated line breaks, hard-wrapped lines, page
furniture and long runs of blank space. Cleaning matters because the
embedding model sees exactly these characters -- noise in the text becomes
noise in the vector, which makes retrieval worse.
"""

import re


def clean_text(text: str) -> str:
    """Normalise raw PDF text into clean, readable prose.

    Steps, in order:
      1. Normalise line endings and remove the NULL / form-feed junk PDFs emit.
      2. Re-join words split across a line break by a hyphen ("inter-\nnational").
      3. Un-wrap hard line breaks inside a sentence, but keep paragraph breaks.
      4. Collapse runs of spaces/tabs and runs of blank lines.
    """
    if not text:
        return ""

    # 1. Normalise line endings, drop control characters PDFs like to insert.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "").replace("\f", "\n\n")

    # 2. "inter-\nnational" -> "international"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 3. A single newline between two normal characters is a PDF line wrap,
    #    not a real paragraph break, so turn it into a space.
    #    Two or more newlines mean a real paragraph break, so leave them.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 4. Collapse whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_meaningful(text: str, min_chars: int = 20) -> bool:
    """True if the text has enough real content to be worth indexing.

    Guards against PDFs that only yield page numbers or whitespace.
    """
    return len(text.strip()) >= min_chars
