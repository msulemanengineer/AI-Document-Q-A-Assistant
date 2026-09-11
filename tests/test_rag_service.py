"""
Tests for the end-to-end RAG pipeline.

A `FakeProvider` stands in for the LLM so these tests are fast, free and
deterministic -- but everything before the LLM (extraction, cleaning,
chunking, embedding, FAISS retrieval, prompt building) is the real code.
"""

import pytest

from backend.services import rag_service as rag_module
from backend.services.llm_service import LLMProvider, MockProvider
from backend.services.rag_service import NOT_FOUND_MESSAGE, RAGService
from backend.utils.errors import EmptyQuestionError, InvalidPDFError, NoDocumentError


class FakeProvider(LLMProvider):
    """Records the prompts it receives and returns a canned answer."""

    name = "fake"

    def __init__(self, reply: str = "A grounded answer [1]."):
        self.reply = reply
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.reply


@pytest.fixture
def service(sample_pdf_bytes) -> RAGService:
    """A RAGService with the sample PDF fully indexed."""
    rag = RAGService()
    rag.index_document(sample_pdf_bytes, "sample.pdf")
    return rag


@pytest.fixture
def fake_llm(monkeypatch) -> FakeProvider:
    """Replace the real provider lookup with a fake, for this test only."""
    provider = FakeProvider()
    monkeypatch.setattr(rag_module, "get_llm_provider", lambda: provider)
    return provider


# ---------------- indexing ----------------


def test_index_document_populates_the_store(service):
    assert service.is_ready()
    assert service.document is not None
    assert service.document.filename == "sample.pdf"
    assert service.document.pages == 3
    assert service.document.chunks > 0
    assert service.vector_store.size() == service.document.chunks


def test_indexing_an_invalid_pdf_raises():
    with pytest.raises(InvalidPDFError):
        RAGService().index_document(b"not a pdf at all", "bad.pdf")


def test_uploading_a_second_document_replaces_the_first(service, sample_pdf_bytes):
    first_store = service.vector_store
    service.index_document(sample_pdf_bytes, "second.pdf")

    assert service.document.filename == "second.pdf"
    assert service.vector_store is not first_store


def test_reset_clears_everything(service):
    service.reset()

    assert not service.is_ready()
    assert service.document is None


# ---------------- retrieval ----------------


def test_retrieval_finds_the_relevant_page(service):
    results = service.retrieve("Where is the Eiffel Tower?", top_k=1)

    assert results
    assert "Eiffel" in results[0].text


def test_retrieval_before_upload_raises():
    with pytest.raises(NoDocumentError):
        RAGService().retrieve("Any question?")


def test_empty_question_raises(service):
    with pytest.raises(EmptyQuestionError):
        service.retrieve("   ")


# ---------------- context building ----------------


def test_build_context_numbers_chunks_and_labels_pages(service):
    results = service.retrieve("Eiffel Tower", top_k=2)
    context = RAGService.build_context(results)

    assert "[1]" in context
    assert "page" in context
    for result in results:
        assert result.text in context


# ---------------- answering ----------------


def test_answer_returns_the_llm_reply_with_sources(service, fake_llm):
    result = service.answer("Where is the Eiffel Tower?")

    assert result.answer == "A grounded answer [1]."
    assert result.found_in_document is True
    assert result.sources
    assert result.provider == "fake"


def test_the_retrieved_context_is_actually_sent_to_the_llm(service, fake_llm):
    result = service.answer("Where is the Eiffel Tower?")

    # The prompt the model saw must contain the retrieved passages and the
    # question -- this is the definition of "retrieval augmented".
    assert "Eiffel" in fake_llm.user_prompt
    assert "Where is the Eiffel Tower?" in fake_llm.user_prompt
    assert result.sources[0].text in fake_llm.user_prompt


def test_the_system_prompt_forbids_inventing_information(service, fake_llm):
    service.answer("Where is the Eiffel Tower?")

    system = fake_llm.system_prompt.lower()
    assert "only" in system
    assert NOT_FOUND_MESSAGE.lower() in system


def test_off_topic_question_retrieves_nothing_above_a_strict_threshold(service):
    # The document says nothing about football. With a strict similarity
    # threshold retrieval returns no chunks at all -- which is what triggers
    # the "not found in the document" answer instead of a guess.
    results = service.vector_store.search(
        rag_module.embedding_service.embed_query("What are the offside rules in football?"),
        top_k=4,
        min_similarity=0.7,
    )

    assert results == []


def test_no_results_means_the_llm_is_not_called(service, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(rag_module, "get_llm_provider", lambda: provider)
    # Force retrieval to return nothing.
    monkeypatch.setattr(service, "retrieve", lambda question, top_k=None: [])

    result = service.answer("anything")

    assert result.answer == NOT_FOUND_MESSAGE
    assert result.found_in_document is False
    assert result.sources == []
    assert provider.user_prompt == ""  # the LLM was never asked


def test_llm_saying_not_found_sets_the_flag(service, monkeypatch):
    monkeypatch.setattr(
        rag_module, "get_llm_provider", lambda: FakeProvider(NOT_FOUND_MESSAGE)
    )
    result = service.answer("Where is the Eiffel Tower?")

    assert result.found_in_document is False


def test_mock_provider_works_without_an_api_key():
    answer = MockProvider().generate("system", "user")

    assert "MOCK MODE" in answer
