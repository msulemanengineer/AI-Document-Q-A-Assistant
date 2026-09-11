"""
Custom exceptions.

Each one maps to a specific, understandable failure the user can act on.
`main.py` converts them into HTTP responses with the right status code,
so the service layer never has to know about HTTP.
"""


class AppError(Exception):
    """Base class for every error we raise on purpose."""

    status_code: int = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidPDFError(AppError):
    """The uploaded file is not a readable PDF (corrupt, encrypted, wrong type)."""

    status_code = 400


class EmptyPDFError(AppError):
    """The PDF opened fine but contains no extractable text (e.g. a scanned image)."""

    status_code = 422


class NoDocumentError(AppError):
    """A question was asked before any document was processed."""

    status_code = 409


class EmptyQuestionError(AppError):
    """The question was blank or whitespace only."""

    status_code = 400


class EmbeddingError(AppError):
    """The embedding model failed to load or failed to encode text."""

    status_code = 500


class VectorStoreError(AppError):
    """FAISS index creation or search failed."""

    status_code = 500


class LLMError(AppError):
    """The LLM provider is misconfigured or the API call failed."""

    status_code = 502
