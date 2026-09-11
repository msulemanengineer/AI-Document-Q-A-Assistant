"""
Tests for the FastAPI endpoints.

`TestClient` calls the app in-process, so no server needs to be running.
The LLM is left in mock mode (the default in .env.example), which means these
tests exercise the real upload -> index -> retrieve path with no API key.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.rag_service import rag_service


@pytest.fixture
def client() -> TestClient:
    # Each test starts from a clean, document-free server.
    rag_service.reset()
    return TestClient(app)


# ---------------- /health and /status ----------------


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["document_loaded"] is False
    assert body["embedding_model"]
    assert body["llm_provider"]


def test_status_reports_no_document_initially(client):
    body = client.get("/status").json()

    assert body["document_loaded"] is False
    assert body["chunks"] == 0


# ---------------- /upload ----------------


def test_upload_indexes_a_valid_pdf(client, sample_pdf_bytes):
    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.pdf"
    assert body["pages"] == 3
    assert body["chunks"] > 0
    assert body["characters"] > 0


def test_health_reflects_the_uploaded_document(client, sample_pdf_bytes):
    client.post("/upload", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    body = client.get("/health").json()

    assert body["document_loaded"] is True
    assert body["filename"] == "sample.pdf"
    assert body["chunks"] > 0


def test_non_pdf_extension_is_rejected(client):
    response = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_corrupt_pdf_is_rejected_with_a_clear_message(client):
    response = client.post(
        "/upload", files={"file": ("broken.pdf", b"%PDF-1.4 garbage", "application/pdf")}
    )

    assert response.status_code == 400
    assert response.json()["error_type"] == "InvalidPDFError"


def test_pdf_with_no_text_returns_422(client, blank_pdf_bytes):
    response = client.post(
        "/upload", files={"file": ("scan.pdf", blank_pdf_bytes, "application/pdf")}
    )

    assert response.status_code == 422
    assert response.json()["error_type"] == "EmptyPDFError"


def test_upload_with_no_file_is_a_validation_error(client):
    assert client.post("/upload").status_code == 422


# ---------------- /ask ----------------


def test_ask_before_upload_returns_409(client):
    response = client.post("/ask", json={"question": "What is this about?"})

    assert response.status_code == 409
    assert response.json()["error_type"] == "NoDocumentError"


def test_ask_returns_an_answer_with_sources(client, sample_pdf_bytes):
    client.post("/upload", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    response = client.post("/ask", json={"question": "Where is the Eiffel Tower?"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Where is the Eiffel Tower?"
    assert body["answer"]
    assert body["sources"]
    # Sources must carry enough information for the user to verify the answer.
    assert "Eiffel" in body["sources"][0]["text"]
    assert body["sources"][0]["page"] == 1
    assert 0.0 <= body["sources"][0]["score"] <= 1.0


def test_top_k_controls_how_many_sources_come_back(client, sample_pdf_bytes):
    client.post("/upload", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    body = client.post("/ask", json={"question": "Paris", "top_k": 1}).json()

    assert len(body["sources"]) <= 1


def test_empty_question_is_rejected_by_validation(client, sample_pdf_bytes):
    client.post("/upload", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})

    # Pydantic's min_length=1 catches this before it reaches the service.
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_whitespace_only_question_is_rejected(client, sample_pdf_bytes):
    client.post("/upload", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["error_type"] == "EmptyQuestionError"


def test_invalid_top_k_is_rejected(client, sample_pdf_bytes):
    client.post("/upload", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})

    assert client.post("/ask", json={"question": "hi", "top_k": 0}).status_code == 422


# ---------------- /reset ----------------


def test_reset_clears_the_document(client, sample_pdf_bytes):
    client.post("/upload", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    response = client.delete("/reset")

    assert response.status_code == 200
    assert response.json()["document_loaded"] is False
    # And asking again is back to the "no document" error.
    assert client.post("/ask", json={"question": "anything"}).status_code == 409
