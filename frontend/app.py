"""
Streamlit frontend.

This is a thin client: it uploads a file, posts a question, and renders the
JSON it gets back. All RAG logic stays in the FastAPI backend, which keeps the
two layers independent -- the API could serve a React app or a CLI just as well.

Run it with:
    streamlit run frontend/app.py
"""

import os
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 180  # seconds; embedding a large PDF can take a while

st.set_page_config(page_title="AI Document Q&A Assistant", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------
# API helpers -- every call returns (data, error_message)
# --------------------------------------------------------------------------


def _read_error(response: requests.Response) -> str:
    """Pull the readable message out of an error response."""
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return f"HTTP {response.status_code}: {response.text[:300]}"


def api_health() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            return None, _read_error(response)
        return response.json(), None
    except requests.exceptions.RequestException:
        return None, (
            f"Cannot reach the backend at {API_URL}. "
            "Start it with:  uvicorn backend.main:app --reload"
        )


def api_upload(filename: str, data: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (filename, data, "application/pdf")},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return None, _read_error(response)
        return response.json(), None
    except requests.exceptions.Timeout:
        return None, "The upload timed out. The PDF may be very large."
    except requests.exceptions.RequestException as exc:
        return None, f"Could not reach the backend: {exc}"


def api_ask(question: str, top_k: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return None, _read_error(response)
        return response.json(), None
    except requests.exceptions.Timeout:
        return None, "The request timed out while waiting for the LLM."
    except requests.exceptions.RequestException as exc:
        return None, f"Could not reach the backend: {exc}"


def api_reset() -> Optional[str]:
    try:
        requests.delete(f"{API_URL}/reset", timeout=10)
        return None
    except requests.exceptions.RequestException as exc:
        return f"Could not reach the backend: {exc}"


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

st.session_state.setdefault("doc_info", None)   # metadata of the indexed PDF
st.session_state.setdefault("result", None)     # last answer from /ask
st.session_state.setdefault("question", "")

# --------------------------------------------------------------------------
# Sidebar: connection, settings, reset
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    health, health_error = api_health()
    if health_error:
        st.error(health_error)
    else:
        st.success("Backend connected")
        st.caption(f"LLM provider: `{health['llm_provider']}`")
        st.caption(f"Embedding model: `{health['embedding_model']}`")
        if health["llm_provider"] == "mock":
            st.warning(
                "Running in **mock mode** — retrieval is real, but no LLM is "
                "called. Set `LLM_PROVIDER` and `LLM_API_KEY` in `.env` for "
                "real answers."
            )

    st.divider()
    top_k = st.slider(
        "Top-K chunks to retrieve",
        min_value=1,
        max_value=10,
        value=4,
        help="How many document passages are sent to the LLM as context. "
        "Higher = more context but more noise and more tokens.",
    )

    st.divider()
    if st.button("🗑️ Clear document", use_container_width=True):
        error = api_reset()
        if error:
            st.error(error)
        else:
            st.session_state.doc_info = None
            st.session_state.result = None
            st.session_state.question = ""
            st.rerun()

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------

st.title("📄 AI Document Q&A Assistant")
st.caption(
    "Upload a PDF and ask questions about it. Answers are generated only from "
    "the document's own content (Retrieval-Augmented Generation)."
)

# ---- Step 1: upload ----
st.subheader("1. Upload a document")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process document", type="primary"):
        with st.spinner("Extracting text, chunking, embedding and indexing..."):
            info, error = api_upload(uploaded_file.name, uploaded_file.getvalue())
        if error:
            st.error(error)
        else:
            st.session_state.doc_info = info
            st.session_state.result = None
            st.success(info["message"])

if st.session_state.doc_info:
    info = st.session_state.doc_info
    st.info(f"**Indexed:** {info['filename']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pages with text", info["pages"])
    col2.metric("Characters", f"{info['characters']:,}")
    col3.metric("Chunks", info["chunks"])
    col4.metric("Chunk size", info["chunk_size"])

st.divider()

# ---- Step 2: ask ----
st.subheader("2. Ask a question")

if not st.session_state.doc_info:
    st.info("Upload and process a PDF first, then you can ask questions about it.")
else:
    question = st.text_input(
        "Your question",
        value=st.session_state.question,
        placeholder="e.g. What are the main findings of this document?",
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            st.session_state.question = question
            with st.spinner("Searching the document and generating an answer..."):
                result, error = api_ask(question, top_k)
            if error:
                st.error(error)
                st.session_state.result = None
            else:
                st.session_state.result = result

# ---- Step 3: answer + sources ----
result = st.session_state.result
if result:
    st.divider()
    st.subheader("Answer")

    if result["found_in_document"]:
        st.markdown(result["answer"])
    else:
        # Being explicit here is the whole point: the system says "I don't know"
        # instead of inventing something.
        st.warning(result["answer"])

    st.caption(
        f"Provider: `{result['provider']}` · "
        f"{len(result['sources'])} chunk(s) retrieved · "
        f"{result['elapsed_seconds']}s"
    )

    if result["sources"]:
        st.subheader("Sources — the passages used to answer")
        st.caption(
            "These are the exact chunks sent to the LLM. Check them to verify "
            "the answer came from your document."
        )
        for position, source in enumerate(result["sources"], start=1):
            with st.expander(
                f"[{position}] Page {source['page']} · similarity {source['score']:.3f}"
            ):
                st.write(source["text"])
    else:
        st.caption("No passage in the document was similar enough to the question.")
