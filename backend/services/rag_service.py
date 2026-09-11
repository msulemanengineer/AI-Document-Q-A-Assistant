"""
The RAG orchestrator. This file ties every other service together.

INDEXING (runs once per uploaded PDF):
    PDF bytes
      -> pdf_service.extract_pages       (extract + clean text)
      -> chunking_service.chunk_document (overlapping chunks)
      -> embedding_service.embed_texts   (chunk vectors)
      -> vector_store.build              (FAISS index)

ANSWERING (runs on every question):
    question
      -> embedding_service.embed_query   (question vector)
      -> vector_store.search             (top-k similar chunks)
      -> build_context                   (numbered context block)
      -> llm_service.generate            (grounded answer)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.services import embedding_service, pdf_service
from backend.services.chunking_service import Chunk, chunk_document
from backend.services.llm_service import get_llm_provider
from backend.services.vector_store import SearchResult, VectorStore
from backend.utils.errors import EmptyQuestionError, NoDocumentError

# --------------------------------------------------------------------------
# PROMPTS -- this is the main defence against hallucination.
# --------------------------------------------------------------------------

# The exact phrase the model must use when the context does not hold the
# answer. Having ONE fixed sentence makes the behaviour easy to detect in
# code and easy to assert in tests.
NOT_FOUND_MESSAGE = "The answer to this question was not found in the uploaded document."

SYSTEM_PROMPT = f"""You are a careful document question-answering assistant.

Follow these rules strictly:
1. Answer ONLY using the document context provided in the user's message.
2. Do NOT use outside knowledge, and do NOT guess or invent any fact,
   number, name or date that is not written in the context.
3. If the context does not contain enough information to answer, reply with
   exactly this sentence and nothing else:
   "{NOT_FOUND_MESSAGE}"
4. Do not pretend to know something that is not in the context. It is better
   to say the information is missing than to give an answer that might be wrong.
5. When you use a passage, cite it inline using its number, like [1] or [2].
6. Keep the answer clear and concise, and quote the document where helpful."""

USER_PROMPT_TEMPLATE = """Document context:
---------------------
{context}
---------------------

Using only the context above, answer this question.

Question: {question}

Answer:"""


@dataclass
class AnswerResult:
    """Everything the API returns for one question."""

    answer: str
    sources: List[SearchResult]
    found_in_document: bool
    provider: str
    elapsed_seconds: float


@dataclass
class DocumentInfo:
    """Metadata about the document currently loaded in memory."""

    filename: str
    pages: int
    characters: int
    chunks: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    indexed_at: float = field(default_factory=time.time)


class RAGService:
    """Holds the index for the current document and answers questions about it.

    NOTE: state lives in memory and there is exactly one document at a time.
    That is a deliberate simplification for a single-user demo; see the
    "Scalability considerations" section of SYSTEM_GUIDE.md for what a real
    deployment would need instead.
    """

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.document: Optional[DocumentInfo] = None

    # ---------------- indexing ----------------

    def index_document(self, pdf_bytes: bytes, filename: str) -> DocumentInfo:
        """Run the full indexing pipeline on one PDF and keep the result."""
        # 1 + 2. Extract and clean the text, page by page.
        pages = pdf_service.extract_pages(pdf_bytes)

        # 3. Split into overlapping chunks that fit the embedding model.
        chunks: List[Chunk] = chunk_document(pages)

        # 4. Turn every chunk into a normalised vector (one batched call).
        vectors = embedding_service.embed_texts([c.text for c in chunks])

        # 5. Build a FRESH FAISS index. Replacing the old one is how uploading
        #    a new PDF cleanly discards the previous document.
        store = VectorStore()
        store.build(chunks, vectors)
        self.vector_store = store

        self.document = DocumentInfo(
            filename=filename,
            pages=len(pages),
            characters=sum(len(text) for _, text in pages),
            chunks=len(chunks),
            embedding_model=settings.EMBEDDING_MODEL,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        return self.document

    # ---------------- retrieval ----------------

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Find the chunks most relevant to the question."""
        if not self.is_ready():
            raise NoDocumentError(
                "No document has been processed yet. Please upload a PDF first."
            )
        if not question or not question.strip():
            raise EmptyQuestionError("Please type a question.")

        query_vector = embedding_service.embed_query(question)
        return self.vector_store.search(query_vector, top_k=top_k)

    @staticmethod
    def build_context(results: List[SearchResult]) -> str:
        """Format retrieved chunks into a numbered block for the prompt.

        The numbers and page labels do two jobs: they let the model cite its
        source as [1], [2], and they let the user check the answer against the
        real document.
        """
        blocks = []
        for position, result in enumerate(results, start=1):
            blocks.append(f"[{position}] (page {result.page})\n{result.text}")
        return "\n\n".join(blocks)

    # ---------------- generation ----------------

    def answer(self, question: str, top_k: Optional[int] = None) -> AnswerResult:
        """Full question -> grounded answer flow."""
        started = time.time()
        results = self.retrieve(question, top_k=top_k)
        provider = get_llm_provider()

        # Short-circuit: if retrieval found nothing above the similarity
        # threshold there is no context to ground an answer in. Calling the LLM
        # here would only invite it to make something up, and would cost a
        # request for nothing.
        if not results:
            return AnswerResult(
                answer=NOT_FOUND_MESSAGE,
                sources=[],
                found_in_document=False,
                provider=provider.name,
                elapsed_seconds=round(time.time() - started, 3),
            )

        context = self.build_context(results)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context, question=question.strip()
        )
        answer_text = provider.generate(SYSTEM_PROMPT, user_prompt)

        if not answer_text:
            answer_text = NOT_FOUND_MESSAGE

        # Did the model itself say the answer is missing? The UI uses this flag
        # to show a warning instead of presenting a non-answer as an answer.
        found = NOT_FOUND_MESSAGE.lower() not in answer_text.lower()

        return AnswerResult(
            answer=answer_text,
            sources=results,
            found_in_document=found,
            provider=provider.name,
            elapsed_seconds=round(time.time() - started, 3),
        )

    # ---------------- state ----------------

    def is_ready(self) -> bool:
        return self.vector_store.is_ready()

    def reset(self) -> None:
        """Forget the current document and free the index."""
        self.vector_store.clear()
        self.document = None

    def status(self) -> Dict[str, Any]:
        return {
            "document_loaded": self.is_ready(),
            "filename": self.document.filename if self.document else None,
            "pages": self.document.pages if self.document else 0,
            "chunks": self.vector_store.size(),
        }


# One shared instance for the API process.
rag_service = RAGService()
