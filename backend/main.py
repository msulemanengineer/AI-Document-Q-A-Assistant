"""
FastAPI application: the HTTP layer.

This file deliberately contains NO machine-learning logic. Its only jobs are
to validate input, call the right service, and turn exceptions into clear HTTP
responses. All the RAG work lives in `backend/services/`.

Run it with:
    uvicorn backend.main:app --reload
Interactive docs:
    http://127.0.0.1:8000/docs
"""

import logging

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request

from backend.config import settings
from backend.models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceChunk,
    StatusResponse,
    UploadResponse,
)
from backend.services.rag_service import rag_service
from backend.utils.errors import AppError, InvalidPDFError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Document Q&A Assistant",
    description=(
        "A Retrieval-Augmented Generation (RAG) API. Upload a PDF, then ask "
        "questions that are answered only from that document."
    ),
    version="1.0.0",
)

# The Streamlit frontend runs on a different port, so the browser treats it as
# a different origin. For a local demo, allowing all origins is fine; a real
# deployment should list the exact frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Error handling: one place that converts our exceptions into HTTP responses.
# --------------------------------------------------------------------------


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("%s: %s", type(exc).__name__, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": type(exc).__name__},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the full traceback for us, return a safe generic message to the user.
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred. Check the server logs.",
            "error_type": "InternalServerError",
        },
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness check plus the current configuration and document state."""
    status = rag_service.status()
    return HealthResponse(
        status="ok",
        llm_provider=settings.LLM_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL,
        document_loaded=status["document_loaded"],
        filename=status["filename"],
        chunks=status["chunks"],
    )


@app.get("/status", response_model=StatusResponse, tags=["system"])
def status() -> StatusResponse:
    """What document, if any, is currently indexed."""
    return StatusResponse(**rag_service.status())


@app.post("/upload", response_model=UploadResponse, tags=["rag"])
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF and run the indexing half of the RAG pipeline.

    extract text -> clean -> chunk -> embed -> build FAISS index
    """
    if not file.filename:
        raise InvalidPDFError("No file was uploaded.")
    if not file.filename.lower().endswith(".pdf"):
        raise InvalidPDFError("Only PDF files are supported.")

    pdf_bytes = await file.read()

    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise InvalidPDFError(
            f"The file is {size_mb:.1f} MB, which is over the "
            f"{settings.MAX_UPLOAD_MB} MB limit."
        )

    logger.info("Indexing %s (%.2f MB)", file.filename, size_mb)
    info = rag_service.index_document(pdf_bytes, file.filename)
    logger.info("Indexed %s: %d pages, %d chunks", info.filename, info.pages, info.chunks)

    return UploadResponse(
        message="Document processed successfully. You can now ask questions.",
        filename=info.filename,
        pages=info.pages,
        characters=info.characters,
        chunks=info.chunks,
        embedding_model=info.embedding_model,
        chunk_size=info.chunk_size,
        chunk_overlap=info.chunk_overlap,
    )


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(request: AskRequest) -> AskResponse:
    """Answer a question using the indexed document.

    embed question -> FAISS search -> build context -> LLM -> grounded answer
    """
    result = rag_service.answer(request.question, top_k=request.top_k)
    return AskResponse(
        question=request.question,
        answer=result.answer,
        found_in_document=result.found_in_document,
        sources=[
            SourceChunk(
                chunk_id=s.chunk_id, page=s.page, score=s.score, text=s.text
            )
            for s in result.sources
        ],
        provider=result.provider,
        elapsed_seconds=result.elapsed_seconds,
    )


@app.delete("/reset", response_model=StatusResponse, tags=["rag"])
def reset() -> StatusResponse:
    """Drop the current document and its index."""
    rag_service.reset()
    return StatusResponse(
        document_loaded=False, message="Document and index cleared."
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT)
