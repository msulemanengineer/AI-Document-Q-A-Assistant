# 📄 AI Document Q&A Assistant

A **Retrieval-Augmented Generation (RAG)** application. Upload a PDF, ask questions
about it in plain English, and get answers that are generated **only from that
document** — with the exact source passages shown next to every answer.

Built with FastAPI, Streamlit, sentence-transformers and FAISS. The RAG pipeline is
written out step by step in plain Python (no LangChain), so every stage is easy to
read, modify and explain.

---

## Quickstart

Needs Python 3.10-3.12. From the project root:

```powershell
py -3.12 -m venv .venv          # see Installation if you don't have 3.12
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then open **two terminals**, both with `.venv` active and both in the project root:

```bash
# terminal 1 - backend
uvicorn backend.main:app --reload

# terminal 2 - frontend
streamlit run frontend/app.py
```

Open <http://localhost:8501>, upload a PDF, click **Process document**, ask a question.

**This works immediately with no API key** — it runs in mock mode, where retrieval is
real and fully functional but the written answer is a placeholder. To get real generated
answers, see [Connecting a real LLM](#connecting-a-real-llm) (takes about two minutes).

---

## Table of contents

- [Project description](#project-description)
- [Why this project exists](#why-this-project-exists)
- [Features](#features)
- [Technology stack](#technology-stack)
- [Architecture](#architecture)
- [The RAG pipeline](#the-rag-pipeline)
- [Folder structure](#folder-structure)
- [Installation](#installation)
- [Environment variables](#environment-variables)
- [Connecting a real LLM](#connecting-a-real-llm)
- [Running the backend](#running-the-backend)
- [Running the frontend](#running-the-frontend)
- [Example usage](#example-usage)
- [API endpoints](#api-endpoints)
- [Running the tests](#running-the-tests)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)

---

## Project description

Large Language Models are good at language but they do not know the contents of
*your* private PDF, and when asked about something they have not seen, they tend to
invent a confident-sounding answer.

This project solves that with RAG. Instead of asking the model to recall facts, we:

1. Split the uploaded PDF into small overlapping passages ("chunks").
2. Convert each chunk into a vector that captures its **meaning** (an embedding).
3. Store those vectors in a **FAISS** index.
4. When a question arrives, convert the question into a vector too, and find the
   most semantically similar chunks.
5. Paste only those chunks into the prompt and ask the LLM to answer **using
   nothing else**.

The result is an answer that is grounded in the document, plus the passages it came
from so the user can check it.

## Why this project exists

This is a portfolio project for an AI/ML internship. It was written to demonstrate,
and to be able to explain in an interview:

- How a complete RAG pipeline works, stage by stage
- Practical use of embeddings and vector similarity search
- Prompt design for reducing hallucination
- Clean separation between ML logic, an HTTP API, and a UI
- Sensible error handling and basic test coverage

A companion file, **[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)**, explains the entire system
in simple English and includes 41 interview questions with answers.

## Features

- 📤 **PDF upload** with validation (type, size, readability)
- ✂️ **Configurable chunking** — chunk size and overlap set in `.env`
- 🧠 **Semantic search** using `all-MiniLM-L6-v2` embeddings + FAISS
- 🎯 **Top-K retrieval** adjustable live from the UI
- 🛡️ **Hallucination control** — a strict system prompt, a similarity threshold, and
  an explicit "not found in the document" answer
- 🔍 **Source citations** — every answer shows the retrieved passages, their page
  numbers and their similarity scores
- 🔌 **Pluggable LLM providers** — OpenAI, Anthropic, any OpenAI-compatible endpoint,
  or a **mock mode that needs no API key**
- ⚡ **FastAPI backend** with automatic interactive docs at `/docs`
- 🖥️ **Streamlit frontend** with progress indicators, error messages and a reset button
- ✅ **Test suite** covering extraction, chunking, embeddings, search, RAG and the API

## Technology stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Async, fast, automatic validation and `/docs` |
| UI | Streamlit | A usable data-app UI in pure Python |
| PDF parsing | pypdf | Pure Python, no system dependencies, installs cleanly on Windows |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | 384-dim, ~80 MB, fast on CPU, strong quality for its size |
| Vector search | FAISS (`IndexFlatIP`) | Exact, fast similarity search; inner product on normalised vectors = cosine similarity |
| LLM | OpenAI / Anthropic / mock | Swappable behind one small interface |
| Validation | Pydantic v2 | Request/response schemas and typed config |
| Config | python-dotenv | Keeps API keys out of the code and out of git |
| Tests | pytest + httpx | Fast in-process API testing |

## Architecture

```
┌──────────────────────┐        HTTP/JSON        ┌────────────────────────────┐
│   Streamlit UI       │ ──────────────────────► │   FastAPI backend          │
│   frontend/app.py    │ ◄────────────────────── │   backend/main.py          │
└──────────────────────┘                         └─────────────┬──────────────┘
                                                               │
                                                 ┌─────────────▼──────────────┐
                                                 │   rag_service.py           │
                                                 │   (orchestrates the flow)  │
                                                 └─────────────┬──────────────┘
                     ┌───────────────┬───────────────┬─────────┴────────┬──────────────┐
                     ▼               ▼               ▼                  ▼              ▼
              pdf_service   chunking_service  embedding_service   vector_store   llm_service
              extract+clean   split+overlap    MiniLM vectors     FAISS index    OpenAI /
                                                                                 Anthropic /
                                                                                 mock
```

The backend contains no UI code and the frontend contains no ML code. `main.py`
handles HTTP only; all the RAG logic lives in `backend/services/`.

## The RAG pipeline

**Indexing** — runs once, when a PDF is uploaded (`POST /upload`):

```
PDF file
  ↓  pdf_service.extract_pages()        extract text page by page
  ↓  text_cleaning.clean_text()         fix hyphenation, unwrap lines, collapse whitespace
  ↓  chunking_service.chunk_document()  split into overlapping chunks (keeps page numbers)
  ↓  embedding_service.embed_texts()    each chunk → a normalised 384-dim vector
  ↓  vector_store.build()               load vectors into a FAISS IndexFlatIP
FAISS index (in memory)
```

**Answering** — runs on every question (`POST /ask`):

```
User question
  ↓  embedding_service.embed_query()    question → a 384-dim vector (same model!)
  ↓  vector_store.search()              cosine similarity against every chunk
  ↓                                     keep Top-K above MIN_SIMILARITY
Top-K relevant chunks
  ↓  rag_service.build_context()        numbered context block with page labels
  ↓  llm_service.generate()             strict system prompt + context + question
Grounded answer + source passages
```

### The parameters that matter

| Parameter | Default | What it controls |
|---|---|---|
| `CHUNK_SIZE` | 1000 chars | **Precision vs. context.** Too small and a chunk loses the surrounding meaning; too large and one vector has to represent several topics at once, which blurs it and makes retrieval imprecise. It must also stay within the embedding model's input limit (256 word-pieces for MiniLM). |
| `CHUNK_OVERLAP` | 200 chars | **Boundary safety.** A fixed cut can separate a statement from its context. Repeating the last 200 characters at the start of the next chunk means an idea that straddles a boundary still appears whole in at least one chunk. Must be smaller than `CHUNK_SIZE`. |
| `TOP_K` | 4 | **Recall vs. noise.** Too small and the passage holding the answer may be missed; too large and the prompt fills with irrelevant text that distracts the model and costs more tokens. |
| `MIN_SIMILARITY` | 0.20 | **The "I don't know" switch.** Chunks scoring below this are discarded. If nothing survives, the app answers "not found in the document" without calling the LLM at all. |

## Folder structure

```
ai-document-qa/
├── backend/
│   ├── main.py                     FastAPI app: routes + error handlers only
│   ├── config.py                   all settings, read from .env
│   ├── models/
│   │   └── schemas.py              Pydantic request/response models
│   ├── services/
│   │   ├── pdf_service.py          PDF → clean text, page by page
│   │   ├── chunking_service.py     text → overlapping chunks
│   │   ├── embedding_service.py    text → normalised vectors
│   │   ├── vector_store.py         FAISS index + similarity search
│   │   ├── llm_service.py          provider abstraction (openai/anthropic/mock)
│   │   └── rag_service.py          orchestrates the whole pipeline + prompts
│   └── utils/
│       ├── errors.py               typed exceptions → HTTP status codes
│       └── text_cleaning.py        PDF text normalisation
├── frontend/
│   └── app.py                      Streamlit UI
├── scripts/
│   └── check_llm.py                verify your .env LLM settings work
├── tests/
│   ├── conftest.py                 builds a real PDF in memory for the tests
│   ├── test_pdf_service.py
│   ├── test_chunking.py
│   ├── test_embedding_and_search.py
│   ├── test_rag_service.py
│   └── test_api.py
├── data/                           scratch space (git-ignored)
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
├── README.md
└── SYSTEM_GUIDE.md                 ← study / viva guide
```

## Installation

**Requirements:** Python **3.10 – 3.12** (3.12 recommended). PyTorch and FAISS do not
yet publish wheels for Python 3.13+, so a newer Python will fail to install with
`Could not find a version that satisfies the requirement torch`.

Check what you have with `py -0p` (Windows) or `python3 --version`. If you only have
3.13+, install 3.12 from [python.org](https://www.python.org/downloads/), or let
[uv](https://docs.astral.sh/uv/) fetch it for you:

```bash
uv venv --python 3.12 .venv      # downloads Python 3.12 if needed
uv pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
git clone <your-repo-url>
cd "AI Document Q&A Assistant"

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
```

If PowerShell blocks the activation script, run this once:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### macOS / Linux

```bash
git clone <your-repo-url>
cd ai-document-qa

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

> **First run note:** the first time you upload a PDF, sentence-transformers
> downloads `all-MiniLM-L6-v2` (~80 MB) from Hugging Face and caches it in
> `~/.cache/huggingface`. That step needs internet; every run after it works offline.
>
> `requirements.txt` installs the default PyTorch build. For a smaller CPU-only
> install, run this **before** `pip install -r requirements.txt`:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Environment variables

Copy `.env.example` to `.env` and edit it. `.env` is git-ignored — **never commit a
real API key.**

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock`, `openai` or `anthropic` |
| `LLM_API_KEY` | *(empty)* | Your API key. Not needed for `mock`. |
| `LLM_MODEL` | *(empty)* | Model name. Blank uses the provider default (`gpt-4o-mini` / `claude-haiku-4-5-20251001`). |
| `LLM_BASE_URL` | *(empty)* | Optional. Point the OpenAI client at Groq, Together, Ollama, etc. |
| `LLM_TEMPERATURE` | `0.0` | 0 keeps answers factual and repeatable. |
| `LLM_MAX_TOKENS` | `600` | Maximum answer length. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model. |
| `CHUNK_SIZE` | `1000` | Characters per chunk. |
| `CHUNK_OVERLAP` | `200` | Characters shared between neighbouring chunks. |
| `TOP_K` | `4` | Chunks retrieved per question. |
| `MIN_SIMILARITY` | `0.20` | Minimum cosine similarity to accept a chunk. |
| `MAX_UPLOAD_MB` | `20` | Upload size limit. |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Where the API listens. |

**Mock mode works with no API key at all.** Retrieval runs for real and the retrieved
passages are shown; only the written answer is replaced by a placeholder.

## Connecting a real LLM

The app ships in **mock mode**, which runs the entire RAG pipeline (extraction,
chunking, embeddings, FAISS retrieval, source citations) with **no API key**. Only the
final written answer is a placeholder. That is enough to develop and demo retrieval —
connect a real LLM when you want generated prose.

### Step 1 — Get an API key

Pick one. Groq is the usual choice for a student project because it has a free tier.

| Provider | Where to get a key | Cost | Notes |
|---|---|---|---|
| **Groq** | <https://console.groq.com/keys> | Free tier | Fast, OpenAI-compatible. Easiest start. |
| **Google Gemini** | <https://aistudio.google.com/apikey> | Free tier | Also OpenAI-compatible. |
| **OpenAI** | <https://platform.openai.com/api-keys> | Paid (needs billing) | Industry standard. |
| **Anthropic** | <https://console.anthropic.com/settings/keys> | Paid (needs billing) | Claude models. |
| **Ollama** | <https://ollama.com> | Free, fully local | No key, no internet, nothing leaves your machine. |

> Free tiers and pricing change — check the provider's current page rather than
> trusting this table.

### Step 2 — Put it in `.env`

Open `.env` (**not** `.env.example`) and edit these four lines. Copy one block:

**Groq** — free, recommended to start
```ini
LLM_PROVIDER=openai
LLM_API_KEY=gsk_your_actual_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1
```

**Google Gemini** — free tier
```ini
LLM_PROVIDER=openai
LLM_API_KEY=your_google_api_key_here
LLM_MODEL=gemini-2.0-flash
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

**OpenAI**
```ini
LLM_PROVIDER=openai
LLM_API_KEY=sk-your_actual_key_here
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=
```

**Anthropic (Claude)**
```ini
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-your_actual_key_here
LLM_MODEL=claude-haiku-4-5-20251001
LLM_BASE_URL=
```

**Ollama** — runs locally, no key, no data leaves your machine
```ini
LLM_PROVIDER=openai
LLM_API_KEY=ollama
LLM_MODEL=llama3.2
LLM_BASE_URL=http://localhost:11434/v1
```
(First run `ollama pull llama3.2` and make sure `ollama serve` is running.
`LLM_API_KEY` must be non-empty but its value is ignored.)

**Rules for editing `.env`:**
- No quotes around values — `LLM_API_KEY=sk-abc123`, not `LLM_API_KEY="sk-abc123"`
- No spaces around the `=`
- No trailing spaces after the key
- Leave `LLM_BASE_URL` **empty** for OpenAI and Anthropic; set it only for
  OpenAI-compatible providers like Groq, Gemini or Ollama

### Step 3 — Verify the key works

Before starting the app, run the built-in checker:

```bash
python scripts/check_llm.py
```

It prints your configuration (with the key masked), builds the provider, and sends one
tiny test prompt. A success looks like:

```
[1/2] Provider built OK  -> openai / llama-3.3-70b-versatile
[2/2] Sending a test prompt...
      Model replied: 'OK'
--------------------------------------------------------------
SUCCESS - your LLM is configured correctly.
```

A bad key looks like this, and tells you what to check:

```
[2/2] Sending a test prompt...
      FAILED: The OpenAI request failed: Error code: 401 - Invalid API Key
```

Fix `.env` and re-run until it says SUCCESS. This is much faster than debugging through
the UI.

### Step 4 — Restart the backend

**`.env` is read once at startup, so you must restart the backend after editing it.**
This is the single most common reason people think their key "didn't work".

Stop the backend with `Ctrl+C`, then:

```bash
uvicorn backend.main:app --reload
```

Confirm it picked up the change:

```bash
curl http://127.0.0.1:8000/health
```

`"llm_provider"` should no longer say `mock`. The Streamlit sidebar shows the same
thing, and its yellow "mock mode" warning disappears.

### Swapping providers later

Change `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL` and `LLM_BASE_URL` in `.env`, then
restart. **No code changes.** That is the point of the provider abstraction in
[backend/services/llm_service.py](backend/services/llm_service.py): every provider
implements one method, `generate(system_prompt, user_prompt) -> str`, and the rest of
the app only ever talks to that interface.

To add a provider that is *not* OpenAI-compatible, write a ~20-line class implementing
that one method and add one line to `get_llm_provider()`.

### Security reminder

`.env` is git-ignored, so your key will not be committed. Never paste a real key into
`.env.example`, the README, a screenshot, or a commit. If you ever expose one, revoke
it in the provider's console immediately.

## Running the backend

With the virtual environment active, **from the project root**:

```bash
uvicorn backend.main:app --reload
```

- API: <http://127.0.0.1:8000>
- Interactive docs: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

## Running the frontend

In a **second terminal**, with the same virtual environment active:

```bash
streamlit run frontend/app.py
```

Then open <http://localhost:8501>. If your API runs somewhere else, set `API_URL`
before starting Streamlit (e.g. `$env:API_URL="http://127.0.0.1:9000"` on Windows).

## Example usage

### In the UI

1. Start the backend, then the frontend.
2. Upload a PDF and click **Process document**. The sidebar shows page, character
   and chunk counts once indexing finishes.
3. Type a question and click **Ask**.
4. Read the answer, then expand **Sources** to see the exact passages and page
   numbers it was built from.

### From the command line

```bash
# 1. Check the API is up
curl http://127.0.0.1:8000/health

# 2. Upload a PDF
curl -X POST http://127.0.0.1:8000/upload -F "file=@mydocument.pdf"

# 3. Ask a question
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is the main conclusion?\", \"top_k\": 4}"
```

A successful `/ask` response looks like this:

```json
{
  "question": "What is the main conclusion?",
  "answer": "The report concludes that ... [1]",
  "found_in_document": true,
  "sources": [
    { "chunk_id": 12, "page": 7, "score": 0.6421, "text": "..." }
  ],
  "provider": "openai",
  "elapsed_seconds": 1.284
}
```

When the document does not contain the answer:

```json
{
  "answer": "The answer to this question was not found in the uploaded document.",
  "found_in_document": false,
  "sources": []
}
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Status, active provider, embedding model, whether a document is loaded |
| `GET` | `/status` | Which document is currently indexed |
| `POST` | `/upload` | Upload a PDF (multipart `file`) and run the indexing pipeline |
| `POST` | `/ask` | Ask a question (`{"question": "...", "top_k": 4}`) |
| `DELETE` | `/reset` | Drop the current document and its index |

### Error responses

Every error returns the same shape — `{"detail": "...", "error_type": "..."}`:

| Status | `error_type` | When |
|---|---|---|
| 400 | `InvalidPDFError` | Not a PDF, corrupt, encrypted, or over the size limit |
| 400 | `EmptyQuestionError` | The question was blank or whitespace |
| 409 | `NoDocumentError` | A question was asked before uploading a document |
| 422 | `EmptyPDFError` | Valid PDF but no extractable text (likely a scan) |
| 422 | *(FastAPI validation)* | Missing file, `top_k` out of range, question too long |
| 500 | `EmbeddingError` / `VectorStoreError` | The model failed to load, or FAISS failed |
| 502 | `LLMError` | Missing API key, or the LLM request failed |

## Running the tests

```bash
pytest
```

The tests build a real PDF in memory, so nothing extra is needed on disk. They cover:

- PDF text extraction, including corrupt and text-free PDFs
- Text cleaning (hyphenation, line unwrapping, whitespace)
- Chunking: sizes, overlap, page tracking, invalid parameters
- Embedding shape, dtype, unit normalisation and semantic ordering
- FAISS search: correct retrieval, `top_k`, score ordering, threshold filtering, error cases
- The full RAG flow with a fake LLM, including "not found" behaviour
- Every API endpoint and its error paths

`test_embedding_and_search.py` and the RAG/API tests load the real embedding model,
so the **first** `pytest` run needs internet and takes a minute or two.

## Limitations

These are real constraints of the current code, not guesses:

- **No OCR.** Scanned or image-only PDFs yield no text and are rejected with a clear
  message.
- **PDF only.** No DOCX, TXT, HTML or Markdown support.
- **One document at a time.** Uploading a new PDF replaces the previous one.
- **In-memory index.** The FAISS index is not saved to disk, so restarting the
  backend loses the indexed document.
- **Not multi-user.** All state is a single module-level object shared by the whole
  process; two users would see each other's document.
- **No conversation memory.** Each question is independent; follow-ups like "and what
  about the second one?" will not resolve.
- **Pure dense retrieval.** Exact identifiers (a part number, a rare acronym) can be
  matched better by keyword search than by embeddings.
- **Table and multi-column layouts** extract poorly, because `pypdf` returns text in
  reading order, not layout order.
- **No authentication or rate limiting.** It is a local demo, not a deployed service.
- **No quantitative evaluation.** Retrieval quality has not been measured against a
  labelled dataset, so this README makes no accuracy claims.

## Future improvements

- Add OCR (Tesseract) so scanned PDFs work
- Persist the FAISS index to disk and support multiple documents with per-user namespaces
- **Hybrid retrieval**: combine BM25 keyword search with dense vectors
- Add a **cross-encoder re-ranker** over the top ~20 candidates for better precision
- Stream the LLM response token by token into the UI
- Add conversation history with query rewriting for follow-up questions
- Build an evaluation set and measure Recall@K, MRR and answer faithfulness
- Swap FAISS for a managed vector database (Qdrant, pgvector) for concurrency and filtering
- Add authentication, per-user rate limiting and Docker packaging

---

## Learning resource

**[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)** explains this entire system in simple English —
what RAG is, what embeddings are, how FAISS works, a walkthrough of every file, plus
**41 interview questions with answers** and ready-to-say 60-second and 2-minute
project explanations.

## License

MIT — free to use and modify.
