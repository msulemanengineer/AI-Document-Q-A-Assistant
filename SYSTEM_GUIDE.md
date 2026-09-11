# SYSTEM GUIDE — AI Document Q&A Assistant

**This is a study guide, not developer documentation.**

It explains the whole project in simple English, from "what is an embedding" all the
way down to individual functions. Read it before an interview or a viva. Every
concept is explained first in plain words, then tied to the exact file and function
in this repository.

---

## Contents

**Part 1 — The idea**
1. [What is this project?](#1-what-is-this-project)
2. [What problem does it solve?](#2-what-problem-does-it-solve)
3. [Why did we build it?](#3-why-did-we-build-it)
4. [What is RAG?](#4-what-is-rag)
5. [Why RAG instead of sending the whole document to the LLM?](#5-why-rag-instead-of-sending-the-whole-document-to-the-llm)

**Part 2 — How it works**
6. [Complete architecture](#6-complete-architecture)
7. [Step-by-step data flow](#7-step-by-step-data-flow)
8. [PDF text extraction](#8-pdf-text-extraction)
9. [What is text chunking?](#9-what-is-text-chunking)
10. [Why do we chunk documents?](#10-why-do-we-chunk-documents)
11. [What is chunk overlap?](#11-what-is-chunk-overlap)
12. [What are embeddings?](#12-what-are-embeddings)
13. [Why are embeddings needed?](#13-why-are-embeddings-needed)
14. [What is sentence-transformers?](#14-what-is-sentence-transformers)
15. [What is FAISS?](#15-what-is-faiss)
16. [What is vector similarity search?](#16-what-is-vector-similarity-search)
17. [Cosine similarity in simple terms](#17-cosine-similarity-in-simple-terms)

**Part 3 — Answering a question**
18. [What happens when the user asks a question?](#18-what-happens-when-the-user-asks-a-question)
19. [How retrieval works](#19-how-retrieval-works)
20. [What is Top-K retrieval?](#20-what-is-top-k-retrieval)
21. [How retrieved context is sent to the LLM](#21-how-retrieved-context-is-sent-to-the-llm)
22. [What is prompt engineering?](#22-what-is-prompt-engineering)
23. [What is hallucination?](#23-what-is-hallucination)
24. [How this project reduces hallucinations](#24-how-this-project-reduces-hallucinations)
25. [What if the answer is not in the document?](#25-what-if-the-answer-is-not-in-the-document)

**Part 4 — The code**
26. [Why FastAPI?](#26-why-fastapi)
27. [Why Streamlit?](#27-why-streamlit)
28. [Every major Python file explained](#28-every-major-python-file-explained)
29. [Every major class and function explained](#29-every-major-class-and-function-explained)
30. [The most important code sections](#30-the-most-important-code-sections)
31. [Example end-to-end request](#31-example-end-to-end-request)

**Part 5 — Practical**
32. [Common errors and how to fix them](#32-common-errors-and-how-to-fix-them)
33. [Limitations of the current system](#33-limitations-of-the-current-system)
34. [Possible future improvements](#34-possible-future-improvements)
35. [Security considerations](#35-security-considerations)
36. [Scalability considerations](#36-scalability-considerations)

**Part 6 — Interview prep**
- [VIVA / INTERVIEW QUESTIONS (41 questions)](#viva--interview-questions)
- [60-SECOND PROJECT EXPLANATION](#60-second-project-explanation)
- [2-MINUTE DEEP EXPLANATION](#2-minute-deep-explanation)

---
---

# PART 1 — THE IDEA

## 1. What is this project?

It is a web application where you **upload a PDF and ask questions about it in plain
English**. The app reads the document, finds the parts that are relevant to your
question, and uses a Large Language Model (LLM) to write an answer based only on
those parts. It also shows you the exact passages it used, with page numbers, so you
can check the answer yourself.

Technically, it is a **Retrieval-Augmented Generation (RAG)** system with three layers:

- a **Streamlit** user interface,
- a **FastAPI** backend that runs the RAG pipeline,
- **sentence-transformers** for embeddings and **FAISS** for vector search.

## 2. What problem does it solve?

Three real problems:

**Problem 1 — LLMs don't know your private documents.**
A model like GPT-4 or Claude was trained on public text up to some cutoff date. It has
never seen your company's annual report, your university handbook, or your lease
agreement. Asked about them, it either refuses or guesses.

**Problem 2 — Reading long documents by hand is slow.**
Finding one number inside a 200-page report means scrolling, using Ctrl+F, and hoping
you guessed the right search word. If the document says "annual turnover" and you
search for "revenue", Ctrl+F finds nothing — even though the answer is right there.

**Problem 3 — LLMs make things up (hallucinate).**
Ask an LLM a question it cannot answer and it will often produce a confident, fluent,
completely invented answer. For anything factual, that is worse than no answer at all.

This project addresses all three: it grounds the model in *your* document, it searches
by meaning rather than by exact words, and it forces the model to say "I don't know"
when the document doesn't contain the answer.

## 3. Why did we build it?

As an AI/ML internship portfolio project, to demonstrate and be able to explain:

- a complete RAG pipeline written stage by stage, not hidden inside a framework,
- practical use of embeddings and vector similarity search,
- prompt design as a real engineering technique for controlling model behaviour,
- clean software structure — ML logic, HTTP API and UI kept separate,
- proper error handling, configuration management, and tests.

RAG is also the single most common way LLMs are actually deployed in industry today,
so understanding it end to end is directly useful.

## 4. What is RAG?

**RAG = Retrieval-Augmented Generation.**

Break the name apart:

- **Retrieval** — search a collection of documents and pull out the pieces relevant to
  the question.
- **Augmented** — add those pieces to the prompt.
- **Generation** — let the LLM write the answer from them.

The one-line version: **instead of asking the model to remember, you give it the
material and ask it to read.**

A useful analogy: a closed-book exam versus an open-book exam.

| | Plain LLM | RAG |
|---|---|---|
| Exam type | Closed book | Open book |
| Source of facts | The model's training data | Your document |
| Unknown topic | Guesses confidently | Can say "not in the document" |
| Updating knowledge | Requires retraining | Upload a new file |
| Can cite sources | No | Yes |

RAG has two phases:

**Phase A — Indexing (once per document):** read the PDF → clean the text → split it
into chunks → embed each chunk → store the vectors in a searchable index.

**Phase B — Querying (every question):** embed the question → find the most similar
chunks → put them in a prompt → generate the answer.

## 5. Why RAG instead of sending the whole document to the LLM?

This is a very common interview question. There are five solid reasons.

**1. Context windows are finite.**
Every model has a maximum input size. A 500-page PDF can easily exceed it, and the
request simply fails.

**2. Cost.**
You pay per token. Sending a 300-page document with every single question means paying
for 300 pages every time. RAG sends about 4 chunks — often less than 1% of the tokens —
so it is dramatically cheaper.

**3. Speed.**
Less input means faster responses. Processing a huge prompt takes seconds; processing
four short passages is nearly instant.

**4. Accuracy — the "lost in the middle" effect.**
Research consistently shows LLMs attend best to information at the beginning and end
of a long context, and can miss facts buried in the middle. Giving the model four
highly relevant passages instead of 300 pages of mostly-irrelevant text measurably
improves the answer.

**5. Scale.**
It doesn't matter how large your document collection is if you only ever retrieve the
top few chunks. "Stuff everything in the prompt" cannot scale past one medium document;
RAG scales to millions.

> **Honest caveat worth saying in an interview:** with very long-context models, simply
> pasting a *short* document into the prompt is a legitimate and simpler approach. RAG
> earns its complexity when documents are large, numerous, frequently updated, or when
> you need citations. Knowing when *not* to use RAG shows real judgement.

---
---

# PART 2 — HOW IT WORKS

## 6. Complete architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER'S BROWSER                             │
│                    (localhost:8501)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STREAMLIT FRONTEND            frontend/app.py                  │
│  Upload widget · question box · answer + sources display        │
│  Contains NO machine-learning code — it only calls the API.     │
└────────────────────────────┬────────────────────────────────────┘
                             │  HTTP + JSON
┌────────────────────────────▼────────────────────────────────────┐
│  FASTAPI BACKEND               backend/main.py                  │
│  POST /upload · POST /ask · GET /health · DELETE /reset         │
│  Validates input, calls services, turns errors into HTTP codes. │
│  Contains NO machine-learning code either.                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  RAG ORCHESTRATOR        backend/services/rag_service.py        │
│  Owns the pipeline order, the prompts, and the document state.  │
└───┬─────────┬──────────┬──────────────┬─────────────┬───────────┘
    │         │          │              │             │
    ▼         ▼          ▼              ▼             ▼
┌────────┐ ┌────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐
│  pdf_  │ │chunking│ │ embedding_ │ │  vector_ │ │    llm_    │
│service │ │_service│ │  service   │ │  store   │ │  service   │
├────────┤ ├────────┤ ├────────────┤ ├──────────┤ ├────────────┤
│ pypdf  │ │ split  │ │ MiniLM     │ │  FAISS   │ │ OpenAI /   │
│ +clean │ │+overlap│ │ 384-dim    │ │IndexFlatIP│ │ Anthropic /│
│        │ │        │ │ normalised │ │ cosine   │ │ mock       │
└────────┘ └────────┘ └────────────┘ └──────────┘ └────────────┘
                                                         │
                                                         ▼
                                                  ┌─────────────┐
                                                  │  LLM  API   │
                                                  │ (external)  │
                                                  └─────────────┘
```

**The key design idea:** each layer knows only about the layer below it.
`main.py` knows nothing about FAISS. `vector_store.py` knows nothing about HTTP.
This is why you could replace Streamlit with React, or FAISS with Qdrant, by changing
one file.

## 7. Step-by-step data flow

### Phase A — Indexing (when a PDF is uploaded)

```
  ┌──────────────┐
  │   PDF file   │   user uploads report.pdf
  └──────┬───────┘
         │  pdf_service.extract_pages()
         ▼
  ┌──────────────┐
  │  Raw text    │   "Total revenue for\nthe fiscal year..."
  │  per page    │   messy: line wraps, hyphens, extra spaces
  └──────┬───────┘
         │  text_cleaning.clean_text()
         ▼
  ┌──────────────┐
  │ Clean text   │   "Total revenue for the fiscal year was 48.6 million euros."
  └──────┬───────┘
         │  chunking_service.chunk_document()
         ▼
  ┌──────────────┐
  │   Chunks     │   Chunk(id=0, page=1, text="Annual Report 2024...")
  │  (with page) │   Chunk(id=1, page=2, text="Financial summary...")
  └──────┬───────┘
         │  embedding_service.embed_texts()
         ▼
  ┌──────────────┐
  │   Vectors    │   [[0.021, -0.114, 0.078, ... ],   ← 384 numbers per chunk
  │ (384-dim)    │    [0.093,  0.052, -0.011, ... ]]  ← each of length 1.0
  └──────┬───────┘
         │  vector_store.build()
         ▼
  ┌──────────────┐
  │ FAISS index  │   ready to search
  └──────────────┘
```

### Phase B — Querying (every question)

```
  ┌──────────────────────┐
  │ "How much revenue?"  │
  └──────────┬───────────┘
             │  embedding_service.embed_query()   ← SAME model as the chunks
             ▼
  ┌──────────────────────┐
  │ Question vector      │   [0.044, -0.102, 0.061, ...]  (384 numbers)
  └──────────┬───────────┘
             │  vector_store.search()
             ▼
  ┌──────────────────────┐
  │ Compare against      │   chunk 0 → 0.255
  │ every chunk vector   │   chunk 1 → 0.581  ← best
  │ (cosine similarity)  │   chunk 2 → 0.198
  └──────────┬───────────┘   chunk 3 → 0.141
             │  keep Top-K, drop anything below MIN_SIMILARITY
             ▼
  ┌──────────────────────┐
  │ Top-K chunks         │   [chunk 1 (page 2, 0.581), chunk 0 (page 1, 0.255)]
  └──────────┬───────────┘
             │  rag_service.build_context()
             ▼
  ┌──────────────────────┐
  │ Numbered context     │   "[1] (page 2)\nFinancial summary. Total revenue..."
  └──────────┬───────────┘
             │  llm_service.generate(SYSTEM_PROMPT, user_prompt)
             ▼
  ┌──────────────────────┐
  │ Grounded answer      │   "Total revenue for 2024 was 48.6 million euros [1]."
  │ + source passages    │   + the chunks, pages and scores shown in the UI
  └──────────────────────┘
```

## 8. PDF text extraction

**File:** [backend/services/pdf_service.py](backend/services/pdf_service.py)

A PDF is a *layout* format, not a text format. Internally it says "draw the glyph 'T'
at coordinate (72, 700)". There is no concept of a paragraph, a sentence, or even
reliably a word. A PDF reader has to reconstruct text from positioned glyphs, which is
why PDF extraction is always a little messy.

We use **pypdf** because it is pure Python — nothing to compile, installs identically
on Windows, macOS and Linux. (PyMuPDF is faster and handles complex layouts better, but
needs a compiled binary and has an AGPL licence.)

Three things this module does that are worth explaining:

**It keeps text per page.** `extract_pages()` returns `[(1, "text..."), (2, "text...")]`
rather than one big string. That page number travels with every chunk all the way to
the final answer, which is what makes "this came from page 7" possible.

**It cleans the text.** Raw PDF text looks like this:

```
Total revenue for the fis-
cal year 2024 was  48.6
million euros.
```

`clean_text()` in [backend/utils/text_cleaning.py](backend/utils/text_cleaning.py)
turns it into:

```
Total revenue for the fiscal year 2024 was 48.6 million euros.
```

It does four things: normalises line endings and strips control characters; rejoins
words split by a hyphen at a line break (`fis-\ncal` → `fiscal`); converts *single*
newlines to spaces (a line wrap) while keeping *double* newlines (a real paragraph
break); and collapses runs of spaces.

**Why cleaning matters:** the embedding model sees exactly these characters. Garbled
text produces a garbled vector, and a garbled vector retrieves the wrong chunk. Text
cleaning is not cosmetic — it directly affects retrieval quality.

**It detects scanned PDFs.** If every page comes back with no meaningful text, the
file is almost certainly a scan (a picture of a page). We raise `EmptyPDFError` with a
message saying OCR would be needed, rather than silently indexing an empty document.

## 9. What is text chunking?

Chunking means **splitting a long document into smaller pieces** that can each be
embedded and retrieved on their own.

```
A 40-page document
        ↓ chunking
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ chunk 0  │ │ chunk 1  │ │ chunk 2  │ │ chunk 3  │ ...
│ ~1000 ch │ │ ~1000 ch │ │ ~1000 ch │ │ ~1000 ch │
│  page 1  │ │  page 1  │ │  page 2  │ │  page 2  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

In this project, chunking happens in
[backend/services/chunking_service.py](backend/services/chunking_service.py).
We measure chunk size in **characters** — simple, predictable, and easy to explain.
(Production systems often count *tokens* instead, which maps more precisely to model
limits.)

Our chunker is slightly smarter than a blind cut: before slicing, it looks backwards a
little for a sentence ending (`. `, `! `, `? `, or a newline) and cuts there instead,
so chunks tend to end on a complete thought. It only searches the last 20% of the
window, so a chunk never ends up drastically shorter than requested.

## 10. Why do we chunk documents?

Four reasons. An interviewer will usually be satisfied with the first two, but knowing
all four is better.

**1. One embedding = one meaning.**
An embedding model compresses its entire input into a *single* fixed-length vector. If
you embed 40 pages into one vector, it becomes the "average meaning" of the whole
document — finance, HR, engineering, all blended into one blurry point. It will match
everything weakly and nothing well. Small chunks each have a single clear topic, so
their vectors are sharp and specific.

**2. Models have hard input limits.**
`all-MiniLM-L6-v2` truncates at 256 word-pieces (roughly 200 words). Anything past that
is **silently discarded** — no error, no warning. Without chunking you would be
indexing only the first paragraph of each document and wouldn't notice.

**3. Precision of retrieval.**
The point of RAG is to hand the LLM exactly the relevant text. If your retrieval unit
is an entire chapter, you are back to sending huge context. Small chunks mean a small,
focused prompt.

**4. Cost and speed.**
Fewer tokens in the prompt means a cheaper and faster request.

**The trade-off** — this is the part worth saying out loud:

| | Small chunks (e.g. 300 chars) | Large chunks (e.g. 2000 chars) |
|---|---|---|
| Retrieval precision | High — vectors are specific | Lower — vectors are blurred |
| Context completeness | Risk of cutting off needed context | More surrounding context kept |
| Number of chunks | Many (slower indexing, more memory) | Few |
| Prompt size | Small | Large |

There is no universally correct value. ~1000 characters with ~200 overlap is a common,
reasonable default for prose documents, and it is what this project uses. Dense
technical text or legal contracts often work better with smaller chunks.

## 11. What is chunk overlap?

Overlap means **neighbouring chunks share some text**: the last N characters of one
chunk are repeated at the start of the next.

Without overlap (chunk size 50, overlap 0):

```
"The company was founded in 2015 in Rotterdam by | Anna Visser and Karel Dijkstra."
 └──────────────── chunk 0 ─────────────────────┘ └────────── chunk 1 ───────────┘
```

Now ask: *"Who founded the company?"*
Chunk 0 has "was founded in 2015" but no names. Chunk 1 has the names but no
indication they are founders. **Neither chunk can answer the question** — the cut
destroyed the link.

With overlap:

```
chunk 0: "The company was founded in 2015 in Rotterdam by Anna Visser"
chunk 1: "in Rotterdam by Anna Visser and Karel Dijkstra."
                └── repeated overlap ──┘
```

Chunk 0 now contains both the fact and the name. The idea survives the boundary.

**The rule:** overlap must be smaller than chunk size — otherwise the sliding window
never moves forward and you get an infinite loop. Our code validates this explicitly
and raises a `ValueError`.

**The cost:** overlap duplicates text, so 20% overlap means roughly 25% more chunks to
store, embed and search. It is a deliberate trade of storage for reliability.

A common heuristic is **10–20% of chunk size**. This project uses 200 of 1000 = 20%.

## 12. What are embeddings?

An **embedding** is a list of numbers that represents the *meaning* of a piece of text.

`all-MiniLM-L6-v2` turns any text into **384 numbers**:

```
"The cat sat on the mat"  →  [0.021, -0.114, 0.078, 0.199, ... ]   (384 values)
```

Those numbers are coordinates in a 384-dimensional space. The crucial property is:

> **Texts with similar meanings get vectors that point in similar directions —
> even if they share no words at all.**

A 2-D sketch of what that space looks like (real space has 384 dimensions, but the
intuition is the same):

```
                    ▲
                    │      • "puppy playing outside"
       ANIMALS      │    • "dog running in the park"
                    │  • "a cat on a mat"
                    │
     ───────────────┼─────────────────────────►
                    │
                    │                  • "quarterly revenue report"
       FINANCE      │                • "total sales for the year"
                    │              • "operating costs were..."
                    ▼
```

Note that "dog running in the park" and "puppy playing outside" share **zero**
significant words, yet they sit right next to each other. That is what makes semantic
search possible — and it is exactly what
`test_similar_sentences_score_higher_than_unrelated_ones` in the test suite verifies.

**Where do embeddings come from?** A neural network (a small transformer) was trained
on millions of sentence pairs with the objective: *make the vectors of related
sentences close together, and unrelated sentences far apart*. The 384 numbers are what
that trained network outputs. Nobody hand-designed them, and no individual dimension
has a human-readable meaning.

## 13. Why are embeddings needed?

Because **keyword search fails on natural questions.**

Suppose the document says:

> "Total revenue for the fiscal year 2024 was 48.6 million euros."

The user asks:

> "How much money did the company make?"

Compare the words:

| User's words | In the document? |
|---|---|
| how | no |
| much | no |
| money | **no** |
| company | no |
| make | no |

Keyword search (Ctrl+F, SQL `LIKE`, basic search) returns **nothing**. The answer is
sitting right there, but not a single meaningful word matches.

Embeddings solve this because "how much money did the company make" and "total revenue
was 48.6 million euros" land close together in vector space — the *meaning* matches
even though the *words* don't. In this project's live test, that exact question
retrieved the revenue passage with a cosine similarity of **0.581**, well above every
other chunk.

This is the difference between **lexical search** (matching characters) and **semantic
search** (matching meaning).

## 14. What is sentence-transformers?

**sentence-transformers** is a Python library that makes it trivial to turn text into
embeddings. The whole usage is:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(["some text", "some other text"])
```

It wraps a Hugging Face transformer model and adds the *pooling* step that turns
per-word outputs into one vector per sentence, plus batching and normalisation.

**Why `all-MiniLM-L6-v2` specifically?**

| Property | Value | Why it matters |
|---|---|---|
| Dimensions | 384 | Small vectors → fast search, low memory |
| Size on disk | ~80 MB | Downloads in seconds, runs on any laptop |
| Layers | 6 | "MiniLM" is a distilled (compressed) model — fast on CPU |
| Max input | 256 word-pieces | Drives our chunk-size choice |
| Speed | ~thousands of sentences/sec on CPU | No GPU needed |
| Quality | Strong for its size | Excellent quality-per-megabyte |

It is the standard sensible default for a CPU-based RAG demo. Larger models
(`all-mpnet-base-v2`, 768-dim, or the BGE/E5 families) score better on benchmarks but
are several times slower and heavier.

**Where in the code:**
[backend/services/embedding_service.py](backend/services/embedding_service.py).
The model is loaded **once** into a module-level variable and reused, because loading
80 MB of weights on every request would be catastrophically slow.

**One critical rule:** the question and the chunks must be embedded by the **same
model**. Two different models produce vectors in two different, incomparable spaces —
comparing them gives meaningless numbers. Our `vector_store.search()` even checks the
dimension and raises a clear error if it changed.

## 15. What is FAISS?

**FAISS = Facebook AI Similarity Search.** It is a C++ library (with Python bindings)
built to find the nearest vectors to a query vector, very fast.

**Why not just use a Python loop?** For 50 chunks, you could. For 50,000 chunks
searched on every question, a NumPy loop becomes the bottleneck. FAISS is written in
optimised C++ with SIMD instructions and is orders of magnitude faster.

**Which index does this project use, and why?**

We use **`IndexFlatIP`**. Read the name in two halves:

- **Flat** — stores every vector and compares the query against *all* of them. This is
  **exact** search: it is guaranteed to return the true nearest neighbours, with 100%
  recall. Other FAISS indexes (`IVF`, `HNSW`, `PQ`) are *approximate* — much faster on
  millions of vectors, but they can miss results. A single PDF produces at most a few
  thousand chunks, where exact search is already instant, so there is no reason to
  accept approximation.
- **IP** — **Inner Product**. Combined with normalised vectors, this gives us cosine
  similarity directly (see the next two sections).

**What FAISS does NOT do:** it does not store your text. It only stores vectors and
returns their *positions* (0, 1, 2...). That is why `VectorStore` keeps
`self.chunks` alongside the index and maps a returned position back to the original
chunk. This is worth remembering — it is a very common interview follow-up.

## 16. What is vector similarity search?

The whole idea in one sentence: **convert everything to vectors, then find the vectors
closest to your query vector.**

```
Stored chunk vectors          Query vector
  chunk 0  •                      ★  "How much revenue?"
  chunk 1  •  ← closest to ★
  chunk 2  •
  chunk 3  •

  measure distance from ★ to every •, return the nearest ones
```

Three common ways to measure "closeness":

| Measure | What it measures | Range | Note |
|---|---|---|---|
| **Cosine similarity** | Angle between vectors | −1 to 1 (higher = more similar) | Ignores length; the standard for text |
| **Dot / inner product** | Angle **and** length | unbounded | Equals cosine *when vectors are normalised* |
| **Euclidean (L2)** | Straight-line distance | 0 upward (lower = more similar) | Sensitive to length |

For text, **cosine is the right choice**, because vector *length* tends to reflect
things like text length rather than meaning, and we only care about meaning — that is,
direction.

**The trick this project uses:** FAISS's `IndexFlatIP` computes inner product, which is
faster than computing cosine directly. But we normalise every vector to length 1 in
`embedding_service.embed_texts()` (`normalize_embeddings=True`). For unit vectors:

```
cosine(a, b) = (a · b) / (|a| × |b|) = (a · b) / (1 × 1) = a · b
```

So **inner product on normalised vectors is exactly cosine similarity.** We get
cosine's correctness with inner product's speed. The test
`test_embeddings_are_normalised_to_unit_length` asserts this property holds, because
if normalisation ever broke, every similarity score in the app would silently become
wrong.

## 17. Cosine similarity in simple terms

Forget the formula for a moment. **Cosine similarity measures whether two arrows point
in the same direction.**

```
   Same direction              Unrelated                 Opposite
   similarity ≈ 1.0         similarity ≈ 0.0         similarity ≈ -1.0

        ↗ ↗                      ↑                        ↗
       (both arrows)             │  →                    (one arrow)
                                 │                        ↙
                             90° apart                 (the other)
```

| Score | Meaning | Example |
|---|---|---|
| 1.0 | Identical meaning | "a car" vs "a car" |
| 0.7 – 0.9 | Very similar | "a car" vs "an automobile" |
| 0.4 – 0.7 | Related | "a car" vs "a motorway" |
| 0.0 – 0.3 | Mostly unrelated | "a car" vs "photosynthesis" |
| below 0 | Opposing directions | rare with modern text embeddings |

The formula, now that the intuition is in place:

```
                    a · b            (a₁b₁ + a₂b₂ + ... + a₃₈₄b₃₈₄)
cosine(a, b)  =  ───────────  =  ────────────────────────────────────
                  |a| × |b|          length of a  ×  length of b
```

The numerator is the dot product; the denominator divides out both lengths, leaving
pure direction.

**Why divide out the length?** So that a long passage and a short one about the same
topic score as similar. Without normalisation, long text would dominate purely by being
long. Cosine cares about *what* the text is about, not *how much* of it there is.

**Real numbers from this project's live run** — document chunks vs. the question
*"How much revenue did the company make?"*:

| Chunk | Topic | Score |
|---|---|---|
| page 2 | "Total revenue for fiscal 2024 was 48.6 million euros..." | **0.581** |
| page 1 | "Northwind Robotics designs warehouse automation..." | 0.255 |
| page 3 | "Research and development... Halo-3 module..." | below threshold |
| page 4 | "Outlook. Second assembly facility in Poland..." | below threshold |

Notice the revenue chunk scores more than double the next best, even though the
question never uses the word "revenue".

---
---

# PART 3 — ANSWERING A QUESTION

## 18. What happens when the user asks a question?

The complete journey of one question, with the exact function at each step:

```
1.  User types "How much revenue did the company make?" and clicks Ask
         │                                    frontend/app.py
         ▼
2.  POST /ask  {"question": "...", "top_k": 4}
         │                                    HTTP
         ▼
3.  Pydantic validates the body (not empty, <=1000 chars, top_k 1-20)
         │                                    backend/models/schemas.py
         ▼
4.  rag_service.answer(question, top_k)
         │                                    backend/services/rag_service.py
         ▼
5.  Guard: is a document loaded?   no  → NoDocumentError   (HTTP 409)
    Guard: is the question blank?  yes → EmptyQuestionError (HTTP 400)
         │
         ▼
6.  embedding_service.embed_query(question)
    → the SAME MiniLM model → 384 normalised numbers
         │
         ▼
7.  vector_store.search(query_vector, top_k, min_similarity)
    → FAISS compares against all chunk vectors
    → returns the k nearest, drops anything below MIN_SIMILARITY
         │
         ├── EMPTY? → return NOT_FOUND_MESSAGE immediately.
         │            The LLM is never called. (see section 25)
         ▼
8.  rag_service.build_context(results)
    → "[1] (page 2)\nFinancial summary. Total revenue...\n\n[2] (page 1)\n..."
         │
         ▼
9.  llm_service.generate(SYSTEM_PROMPT, user_prompt)
    → system prompt: "answer ONLY from the context, never invent..."
    → user prompt:   the context block + the question
         │
         ▼
10. Check the reply: did the model use the fallback sentence? → found_in_document
         │
         ▼
11. AskResponse JSON: answer, found_in_document, sources[], provider, elapsed_seconds
         │
         ▼
12. Streamlit renders the answer, then the source passages in expanders
    with page numbers and similarity scores.
```

## 19. How retrieval works

Retrieval is steps 6 and 7 above. Mechanically:

**Step 1 — Embed the question.** The question goes through the *same* model as the
chunks, producing one 384-dimensional normalised vector.

```python
query_vector = embedding_service.embed_query(question)   # shape (1, 384)
```

**Step 2 — Compare against every chunk.** FAISS computes the inner product (= cosine
similarity, because everything is normalised) between the query vector and all stored
chunk vectors, then returns the highest-scoring ones.

```python
scores, indices = self.index.search(query_vector, k)
# scores  = [[0.581, 0.255, 0.198, 0.141]]
# indices = [[1,     0,     2,     3    ]]   <- positions, not text
```

**Step 3 — Map positions back to text.** FAISS returned positions. `VectorStore` looks
each one up in `self.chunks` to recover the text, page number and chunk id.

**Step 4 — Filter by threshold.** Anything scoring below `MIN_SIMILARITY` (default
0.20) is discarded. This is what prevents the system from dredging up a weakly related
paragraph when the document simply does not discuss the topic.

Two small but important robustness details in the code:

- `k = min(top_k, self.index.ntotal)` — never ask FAISS for more neighbours than it
  holds. (FAISS pads missing results with index `-1`; we skip those too.)
- A dimension check — if the query vector's length does not match the index, we raise a
  clear "the embedding model changed, please re-upload" error instead of crashing.

## 20. What is Top-K retrieval?

**K** is simply *how many chunks to retrieve*. `TOP_K = 4` means "give me the 4 most
similar passages".

It is a **recall vs. noise** trade-off:

```
K too small (K=1)                       K too large (K=20)
─────────────────                       ──────────────────
✗ If the top chunk is not the one        ✗ 16 irrelevant chunks distract the model
  holding the answer, you fail           ✗ Prompt becomes large and expensive
✓ Tiny, cheap prompt                     ✗ "Lost in the middle" — the good chunk is
✓ Zero irrelevant text                     buried among noise
                                         ✓ Very unlikely to miss the answer
```

Why is K=1 risky? Because an answer is often spread across several chunks — a
definition in one, an example in another — and because the highest-scoring chunk is not
always the most *useful* one. A little redundancy is cheap insurance.

**K = 3-5 is the common sweet spot** for chunks of ~1000 characters. This project
defaults to 4 and exposes a slider in the Streamlit sidebar, so you can demonstrate the
trade-off live in an interview — which is a nice thing to be able to do.

## 21. How retrieved context is sent to the LLM

The retrieved chunks are formatted into a numbered block by
`RAGService.build_context()`:

```
[1] (page 2)
Financial summary. Total revenue for the fiscal year 2024 was 48.6 million
euros, an increase of 22 percent over the previous year...

[2] (page 1)
Annual Report 2024. Northwind Robotics designs warehouse automation systems...
```

The numbering does two jobs: it gives the model a handle to cite (`[1]`), and it lets
the user match a citation back to the passage shown in the UI.

That block is then dropped into the user prompt template:

```
Document context:
---------------------
{context}
---------------------

Using only the context above, answer this question.

Question: {question}

Answer:
```

and sent together with the system prompt. Three deliberate choices here:

**The delimiters (`---------------------`).** They mark clearly where the document text
starts and stops, so the model does not confuse document content with instructions.

**Context comes before the question.** The model reads the material first, then the
task — the same order that works for a human.

**System and user prompts are separate.** The system prompt carries the *rules* (never
invent, say when you do not know); the user prompt carries the *data* (context +
question). Models are trained to weight system instructions heavily, so rules placed
there are followed more reliably.

## 22. What is prompt engineering?

**Prompt engineering is designing the text you send to an LLM so that it behaves the
way you need it to.** You cannot change the model's weights, but you can change what it
reads — and that alone changes the output dramatically.

Compare:

| Weak prompt | Strong prompt (what we use) |
|---|---|
| "Answer this question: {q}" | "Answer ONLY using the provided context. Do not use outside knowledge. If the context is insufficient, reply with exactly: '...'. Cite passages as [1], [2]." |

The weak version invites the model to answer from memory. The strong one constrains it.

This project's system prompt (in
[backend/services/rag_service.py](backend/services/rag_service.py)) uses six named
techniques:

1. **Role assignment** — "You are a careful document question-answering assistant."
   Sets the behavioural frame.
2. **Explicit scope restriction** — "Answer ONLY using the document context."
   The core grounding instruction.
3. **Negative constraints** — "Do NOT use outside knowledge... do NOT guess or invent."
   Stating what *not* to do is as important as stating what to do.
4. **A fixed fallback string** — the model is told the *exact* sentence to use when it
   cannot answer. One deterministic phrase is far easier to detect in code and to
   assert in tests than a hundred paraphrases of "I do not know".
5. **Citation instruction** — "cite using [1], [2]" — makes answers traceable.
6. **Explicit permission to fail** — "It is better to say the information is missing
   than to give an answer that might be wrong." Models default to being helpful; you
   have to actively tell them that admitting ignorance is the preferred behaviour.

We also set **`temperature = 0.0`**. Temperature controls randomness: higher values
make the model sample more creatively. For RAG you want the opposite — faithful,
repeatable extraction from the given text — so 0 is the right setting.

## 23. What is hallucination?

A **hallucination** is when an LLM produces information that sounds confident and
fluent but is **factually wrong or entirely invented**.

Examples:

- Inventing a statistic: "Revenue grew by 34%" when the document says 22%.
- Inventing a citation: naming a paper or an author that does not exist.
- Answering from training data instead of the document, and presenting it as if it came
  from the document.
- Filling a gap: the document never states the founding year, so the model supplies a
  plausible-looking one.

**Why does it happen?** An LLM is fundamentally a next-token predictor. It was trained
to produce text that *looks like* good text — not to verify facts. It has no internal
"do I actually know this?" signal, and it was trained on data where a question is
almost always followed by an answer, never by "I do not know". So when it lacks the
information, its learned behaviour is still to produce a fluent, answer-shaped response.

**Why it matters:** a wrong answer delivered confidently is more dangerous than no
answer, because the user has no signal telling them to distrust it.

## 24. How this project reduces hallucinations

Five defences, arranged in layers. **Be clear in an interview that these *reduce*
hallucination — no system eliminates it.**

**Defence 1 — Grounding (the biggest one).**
The model is given the actual relevant text. Most hallucination comes from the model
having to rely on fuzzy memory; supplying the source text removes that need.

**Defence 2 — A strict system prompt.**
"Answer ONLY using the provided context. Do NOT use outside knowledge. Do not guess or
invent any fact, number, name or date." Explicit and repeated.

**Defence 3 — An explicit, exact escape hatch.**
The model is told the precise sentence to use when the answer is not there:
*"The answer to this question was not found in the uploaded document."*
Giving the model a concrete, permitted way to fail makes it far more likely to use it
than to invent something.

**Defence 4 — The similarity threshold (a code-level guarantee, not a request).**
This is the strongest defence, because it does not depend on the model cooperating. If
no chunk scores above `MIN_SIMILARITY`, `rag_service.answer()` returns the "not found"
message **without calling the LLM at all**. A model that is never invoked cannot
hallucinate. Prompt instructions are a request; this is enforcement.

```python
if not results:
    return AnswerResult(answer=NOT_FOUND_MESSAGE, sources=[],
                        found_in_document=False, ...)
```

The test `test_no_results_means_the_llm_is_not_called` asserts exactly this.

**Defence 5 — Source transparency.**
Every answer ships with the passages it was built from, their page numbers and their
similarity scores. This does not prevent hallucination, but it makes it *detectable* —
the user can verify any claim in seconds. Verifiability is a legitimate and important
part of a hallucination strategy.

**Supporting choice:** `temperature = 0.0` minimises creative drift.

**What this project does NOT do** (worth acknowledging — it shows you know the limits):
it does not verify the answer against the context with a second model call, it does not
check that citation numbers are valid, and it cannot stop a model from subtly
misreading a passage it was correctly given.

## 25. What if the answer is not in the document?

The system says so, explicitly. There are two separate paths, which is worth explaining
clearly because it demonstrates defence in depth.

**Path A — Retrieval finds nothing (handled in code).**

The question is about a topic the document never covers, so every chunk scores below
`MIN_SIMILARITY`. `vector_store.search()` returns an empty list and
`rag_service.answer()` short-circuits: it returns `NOT_FOUND_MESSAGE` with
`found_in_document = False` and an empty `sources` list, **without calling the LLM**.
That saves an API call and removes any opportunity to hallucinate.

Verified live in this project — asking *"What is the recipe for chocolate cake?"* of a
robotics annual report returned:

```json
{ "found_in_document": false, "sources": [] }
```

**Path B — Retrieval finds something, but it does not answer the question.**

Some chunks are topically related enough to pass the threshold, but none actually
contains the answer. Here the LLM *is* called, and the system prompt instructs it to
reply with the exact fallback sentence. `rag_service` then checks the reply:

```python
found = NOT_FOUND_MESSAGE.lower() not in answer_text.lower()
```

and sets `found_in_document` accordingly.

**In the UI**, both cases look the same to the user: the answer is rendered as a
**yellow warning box** instead of normal text, so a non-answer is never visually
presented as an answer.

---
---

# PART 4 — THE CODE

## 26. Why FastAPI?

FastAPI is a modern Python web framework for building APIs. Reasons it fits here:

**1. Automatic validation from type hints.** You declare a Pydantic model and FastAPI
validates every incoming request against it for free:

```python
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
```

An empty question or `top_k=0` is rejected with a clear 422 before your code runs. In
Flask you would write all those checks by hand.

**2. Automatic interactive documentation.** FastAPI generates an OpenAPI schema and
serves a live, clickable API explorer at `/docs`. You can upload a PDF and ask a
question from the browser with no frontend at all. Excellent for a demo.

**3. Async support.** Built on ASGI, so `async def` endpoints handle concurrent
requests without blocking. Our `/upload` is `async` because reading the uploaded file
is I/O.

**4. Speed.** Built on Starlette and Uvicorn; among the fastest Python frameworks.

**5. Clean error handling.** A single `@app.exception_handler(AppError)` converts every
custom exception into the right HTTP status code, so no route needs try/except.

**Alternatives and why not:** *Flask* — mature, but no built-in validation, no async, no
auto-docs. *Django* — a full web framework with an ORM and templates; far more than an
API needs. *Streamlit alone* — possible, but then the ML logic is welded to the UI and
cannot be reused by anything else.

## 27. Why Streamlit?

Streamlit turns a Python script into a web app. No HTML, no CSS, no JavaScript.

```python
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
question = st.text_input("Your question")
if st.button("Ask"):
    st.markdown(answer)
```

That is a working upload + question UI in five lines.

**Why it fits this project:**

- **Speed of development.** The focus is the RAG pipeline, not CSS.
- **One language.** No context switch to a JS toolchain.
- **Built-in widgets that fit ML demos exactly** — file uploaders, sliders, spinners,
  expanders, metrics. The Top-K slider in the sidebar is one line.
- **Standard in the ML community.** Interviewers recognise it immediately as the normal
  way to demo a model.

**Honest limitations** (worth mentioning — it shows judgement): Streamlit re-runs the
entire script top to bottom on every interaction, which is why state must be kept in
`st.session_state`. It is single-page and not designed for multi-user production apps.
For a real product you would build a React/Next.js frontend against the same FastAPI
backend — which is exactly why the backend contains no UI code.

## 28. Every major Python file explained

### `backend/config.py`
All settings in one place, read from environment variables via `python-dotenv`.
Defines `PROJECT_ROOT`, loads `.env`, and exposes a single `settings` object holding the
LLM provider, embedding model name, chunk size, overlap, top-k, similarity threshold,
upload limit and server host/port. Helpers `_get_int` and `_get_float` fall back to a
default if an env var is missing or malformed, so a typo in `.env` cannot crash startup.
**Why it exists:** no magic numbers scattered through the code, and no secret ever
hardcoded.

### `backend/utils/errors.py`
A small exception hierarchy. `AppError` is the base and carries a `status_code`.
Subclasses: `InvalidPDFError` (400), `EmptyPDFError` (422), `NoDocumentError` (409),
`EmptyQuestionError` (400), `EmbeddingError` (500), `VectorStoreError` (500),
`LLMError` (502).
**Why it exists:** the service layer raises meaningful, HTTP-agnostic exceptions;
`main.py` maps them to status codes in one place. Services never import FastAPI.

### `backend/utils/text_cleaning.py`
`clean_text()` normalises raw PDF text: line endings, control characters, hyphenated
line breaks, hard line wraps, whitespace runs. `is_meaningful()` decides whether a page
has enough content to index.
**Why it exists:** the embedding model sees these exact characters, so text quality
directly determines retrieval quality.

### `backend/services/pdf_service.py`  — pipeline step 1-2
`extract_pages()` opens the PDF with `pypdf`, handles encryption, extracts text page by
page, cleans it, and drops empty pages. Raises `InvalidPDFError` for unreadable files
and `EmptyPDFError` when nothing extractable is found (the scanned-PDF case).
`extract_text()` and `count_pages()` are small helpers.

### `backend/services/chunking_service.py`  — pipeline step 3
`Chunk` is a dataclass holding `chunk_id`, `text` and `page`. `_split_page()` slides a
`chunk_size` window forward by `chunk_size - chunk_overlap`, preferring to cut at a
sentence boundary. `chunk_document()` runs it per page so page numbers survive, and
validates the parameters.

### `backend/services/embedding_service.py`  — pipeline step 4
Lazily loads `all-MiniLM-L6-v2` once into a module-level variable. `embed_texts()`
encodes a batch into a normalised float32 array; `embed_query()` encodes one question;
`get_dimension()` reports the vector length. All failures become `EmbeddingError`.

### `backend/services/vector_store.py`  — pipeline steps 5 and 8
The `VectorStore` class wraps a FAISS `IndexFlatIP` plus the chunk list. `build()`
creates the index, `search()` runs similarity search with top-k and threshold filtering
and maps FAISS positions back to chunks, and `is_ready()`/`size()`/`clear()` manage
state. `SearchResult` is the returned dataclass (chunk_id, text, page, score).

### `backend/services/llm_service.py`  — pipeline step 7
The provider abstraction. `LLMProvider` is an ABC with one method,
`generate(system_prompt, user_prompt) -> str`. `MockProvider` needs no key,
`OpenAIProvider` covers OpenAI and any OpenAI-compatible endpoint via `LLM_BASE_URL`,
`AnthropicProvider` covers Claude. `get_llm_provider()` builds the one named in `.env`
and caches it.

### `backend/services/rag_service.py`  — the orchestrator
Holds `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE` and `NOT_FOUND_MESSAGE`. The `RAGService`
class owns the current document and index and exposes `index_document()`, `retrieve()`,
`build_context()`, `answer()`, `reset()` and `status()`. A module-level `rag_service`
instance is shared by the API process.
**This is the file to read first** if you want to understand the project.

### `backend/models/schemas.py`
Pydantic request/response models: `AskRequest`, `UploadResponse`, `AskResponse`,
`SourceChunk`, `HealthResponse`, `StatusResponse`, `ErrorResponse`. These drive
validation, serialisation and the `/docs` page.

### `backend/main.py`
The FastAPI app. CORS middleware, two exception handlers, and five endpoints:
`GET /health`, `GET /status`, `POST /upload`, `POST /ask`, `DELETE /reset`.
Deliberately contains **no ML logic**.

### `frontend/app.py`
The Streamlit UI. Four API helper functions that each return `(data, error)`, session
state for the indexed document and the last answer, a sidebar (health, Top-K slider,
clear button), and the main flow: upload → process → ask → answer + sources.

### `tests/conftest.py`
Builds a valid multi-page PDF **in memory**, byte by byte, so the tests need no fixture
files on disk and no extra dependency. Provides `sample_pdf_bytes` (3 pages of known
text) and `blank_pdf_bytes` (valid PDF, no text).

## 29. Every major class and function explained

### `pdf_service.extract_pages(pdf_bytes) -> List[Tuple[int, str]]`
Opens the PDF, decrypts if it uses an empty password, iterates pages, extracts and
cleans each one, and keeps only pages with meaningful text. Returns `(page_number, text)`
pairs with 1-based page numbers. A single broken page is caught and skipped rather than
failing the whole upload. Raises `InvalidPDFError` or `EmptyPDFError`.

### `text_cleaning.clean_text(text) -> str`
Four ordered steps: normalise line endings and strip `\x00`/`\f`; rejoin hyphenated
line breaks with a regex; convert single newlines to spaces while preserving double
newlines; collapse whitespace runs. The regex `(?<!\n)\n(?!\n)` is the clever bit — it
matches a newline that is *not* adjacent to another newline, i.e. a line wrap rather
than a paragraph break.

### `chunking_service._split_page(text, chunk_size, chunk_overlap) -> List[str]`
The sliding window. Starts at 0, takes `chunk_size` characters, then looks backwards
through the last 20% of the window for `. `, `! `, `? ` or `\n` and cuts there instead
if one is found. Advances by `max(start + step, end - chunk_overlap)`, which honours the
overlap even when the boundary search moved the cut backwards.

### `chunking_service.chunk_document(pages, chunk_size, chunk_overlap) -> List[Chunk]`
Validates the parameters (`chunk_size > 0`, `0 <= overlap < chunk_size`), then splits
each page and assigns sequential `chunk_id`s while carrying the page number through.
Uses `is None` rather than `or` for defaults, so an explicit `0` is rejected instead of
silently replaced.

### `embedding_service.get_model(model_name) -> SentenceTransformer`
Loads the model once and caches it in a module-level variable, reloading only if the
model name changes. `SentenceTransformer` is imported *inside* the function because
importing it pulls in torch and takes seconds.

### `embedding_service.embed_texts(texts, batch_size=32) -> np.ndarray`
Validates the input, calls `model.encode(..., normalize_embeddings=True)`, and returns
`float32` (required by FAISS). The normalisation is the single most important line in
the file — it is what makes inner product equal cosine similarity.

### `VectorStore.build(chunks, vectors) -> None`
Checks that there is exactly one vector per chunk, creates `faiss.IndexFlatIP(dim)`,
adds the vectors, and stores the chunk list. On any failure it clears itself so the
object is never left half-built.

### `VectorStore.search(query_vector, top_k, min_similarity) -> List[SearchResult]`
The retrieval core. Guards against an empty index and a dimension mismatch, clamps
`k` to the index size, calls `self.index.search()`, skips FAISS's `-1` padding, filters
by `min_similarity`, and maps each surviving position back to its `Chunk` to build a
`SearchResult`.

### `LLMProvider.generate(system_prompt, user_prompt) -> str`
The whole provider interface — one method. This is why swapping providers is a
one-line change in `.env`. `OpenAIProvider` sends the system prompt as a `system`
*message*; `AnthropicProvider` passes it as a separate `system` *parameter* — one of the
few real API differences, and a good detail to know.

### `RAGService.index_document(pdf_bytes, filename) -> DocumentInfo`
The indexing pipeline in five readable lines: extract, chunk, embed, build a **fresh**
`VectorStore`, record metadata. Building a new store (rather than mutating the old one)
is how uploading a second PDF cleanly discards the first.

### `RAGService.retrieve(question, top_k) -> List[SearchResult]`
Guards (`NoDocumentError`, `EmptyQuestionError`), embeds the question, searches.

### `RAGService.build_context(results) -> str`
Formats results as `[1] (page 2)\n<text>` blocks joined by blank lines.

### `RAGService.answer(question, top_k) -> AnswerResult`
The full query flow, including the short-circuit when retrieval returns nothing and the
`found_in_document` check on the model's reply. Returns answer, sources, the flag, the
provider name and elapsed time.

### `main.upload(file)` and `main.ask(request)`
Thin HTTP wrappers. `upload` checks the filename ends in `.pdf` and the size is within
`MAX_UPLOAD_MB`, then calls `index_document`. `ask` calls `answer` and converts
`SearchResult` dataclasses into `SourceChunk` Pydantic models.

### `main.app_error_handler(request, exc)`
One function that turns any `AppError` into
`{"detail": ..., "error_type": ...}` with the exception's own `status_code`.

## 30. The most important code sections

If an interviewer says "show me the interesting part", these are the five to open.

### (a) Normalisation — why cosine works
`backend/services/embedding_service.py`

```python
vectors = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,   # <-- makes every vector length 1
    show_progress_bar=False,
)
return np.asarray(vectors, dtype="float32")   # FAISS requires float32
```

Scaling every vector to unit length means `dot(a, b) == cosine(a, b)`, so FAISS's fast
inner-product index gives exact cosine similarity.

### (b) The FAISS index choice
`backend/services/vector_store.py`

```python
self.index = faiss.IndexFlatIP(self.dimension)
self.index.add(vectors)
```

Flat = exact search, 100% recall. IP = inner product, which is cosine here. Two lines
build the entire vector database.

### (c) Mapping FAISS positions back to text
`backend/services/vector_store.py`

```python
scores, indices = self.index.search(query_vector, k)
for score, idx in zip(scores[0], indices[0]):
    if idx < 0:                      # FAISS pads with -1
        continue
    if float(score) < min_similarity:   # the "I don't know" filter
        continue
    chunk = self.chunks[int(idx)]    # position -> original text + page
```

FAISS stores no text; this loop is where numbers become readable passages again.

### (d) The anti-hallucination system prompt
`backend/services/rag_service.py`

```python
NOT_FOUND_MESSAGE = "The answer to this question was not found in the uploaded document."

SYSTEM_PROMPT = f"""You are a careful document question-answering assistant.
...
3. If the context does not contain enough information to answer, reply with
   exactly this sentence and nothing else:
   "{NOT_FOUND_MESSAGE}"
..."""
```

One constant, referenced in the prompt, checked in the code and asserted in the tests.

### (e) The short-circuit that makes hallucination impossible
`backend/services/rag_service.py`

```python
results = self.retrieve(question, top_k=top_k)
...
if not results:
    return AnswerResult(answer=NOT_FOUND_MESSAGE, sources=[],
                        found_in_document=False, ...)
```

If nothing is relevant, the LLM is never called. This is the difference between
*asking* a model to behave and *guaranteeing* it.

## 31. Example end-to-end request

A real trace from this project, running against a 4-page annual report.

**1. Start the backend**

```
uvicorn backend.main:app --reload
```

**2. Upload the PDF**

```
POST /upload      (multipart, file=northwind_report.pdf)
```

Server-side: `extract_pages` returns 4 pages, 785 characters of clean text;
`chunk_document` produces 4 chunks (each page is short enough to be one chunk);
`embed_texts` produces a `(4, 384)` float32 array; `VectorStore.build` creates an
`IndexFlatIP(384)` holding 4 vectors.

```json
{
  "message": "Document processed successfully. You can now ask questions.",
  "filename": "northwind_report.pdf",
  "pages": 4, "characters": 785, "chunks": 4,
  "embedding_model": "all-MiniLM-L6-v2",
  "chunk_size": 1000, "chunk_overlap": 200
}
```

**3. Ask a question**

```json
POST /ask   { "question": "How much revenue did the company make?", "top_k": 2 }
```

Server-side: the question becomes a `(1, 384)` vector; FAISS scores all 4 chunks; two
pass the 0.20 threshold.

```json
{
  "found_in_document": true,
  "sources": [
    { "page": 2, "score": 0.581,
      "text": "Financial summary. Total revenue for the fiscal year 2024 was 48.6 million euros..." },
    { "page": 1, "score": 0.255,
      "text": "Annual Report 2024. Northwind Robotics designs warehouse automation systems..." }
  ]
}
```

The revenue passage wins by more than 2x — **even though the question never contains the
word "revenue"**. That single fact is the clearest demonstration of semantic search
working, and it is a good thing to point at in an interview.

**4. The prompt the LLM actually receives**

```
Document context:
---------------------
[1] (page 2)
Financial summary. Total revenue for the fiscal year 2024 was 48.6 million euros,
an increase of 22 percent over the previous year. Operating costs were 31.2 million
euros. The company employed 412 people at the end of December 2024.

[2] (page 1)
Annual Report 2024. Northwind Robotics designs warehouse automation systems. The
company was founded in 2015 in Rotterdam by Anna Visser and Karel Dijkstra.
---------------------

Using only the context above, answer this question.

Question: How much revenue did the company make?

Answer:
```

**5. Ask something the document does not cover**

```json
POST /ask   { "question": "What is the recipe for chocolate cake?" }
```

No chunk clears the threshold, so the LLM is never called:

```json
{
  "answer": "The answer to this question was not found in the uploaded document.",
  "found_in_document": false,
  "sources": []
}
```

---
---

# PART 5 — PRACTICAL

## 32. Common errors and how to fix them

### Installation

**`ERROR: Could not find a version that satisfies the requirement torch`**
You are on Python 3.13 or newer. PyTorch and FAISS do not publish wheels for it yet.
Create the virtual environment with Python 3.10-3.12:
`py -3.12 -m venv .venv`

**`ModuleNotFoundError: No module named 'backend'`**
You ran the command from the wrong directory, or the virtual environment is not
active. Run everything **from the project root**, with `.venv` activated.

**PowerShell: "running scripts is disabled on this system"**
Windows blocks the activation script by default. Run once, in that terminal:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**`No module named pip` inside the venv**
The venv was created without pip (some tools do this). Either recreate it with
`python -m venv .venv`, or run `python -m ensurepip --upgrade`.

### First run

**It hangs for a minute on the first upload**
Expected. sentence-transformers is downloading `all-MiniLM-L6-v2` (~80 MB) from
Hugging Face. It is cached in `~/.cache/huggingface` and every later run is fast.

**`EmbeddingError: Could not load the embedding model`**
No internet on the first run, or a firewall is blocking Hugging Face. Connect once to
let the model download; after that the app works offline.

**Windows warning: "huggingface_hub cache-system uses symlinks by default..."**
Harmless. Windows needs Developer Mode for symlinks; without it the cache just uses a
little more disk. Silence it with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

### Using the app

**"Cannot reach the backend at http://127.0.0.1:8000"**
The backend is not running, or it is on another port. Start it in a second terminal:
`uvicorn backend.main:app --reload`. If you changed the port, set `API_URL` before
starting Streamlit.

**HTTP 422 `EmptyPDFError` — "No readable text was found in this PDF"**
The PDF is a scan — a picture of a page, with no text layer. This app has no OCR. Use a
text-based PDF, or add OCR (see section 34).

**HTTP 400 `InvalidPDFError` — "This PDF is password protected"**
Remove the password and re-upload.

**HTTP 409 `NoDocumentError` — "No document has been processed yet"**
You asked a question before uploading, or the backend restarted (the index is in
memory, so a restart loses it). Upload again.

**HTTP 502 `LLMError` — "LLM_API_KEY is not set"**
You set `LLM_PROVIDER=openai` (or `anthropic`) without a key. Either add
`LLM_API_KEY=...` to `.env`, or set `LLM_PROVIDER=mock` to run without an LLM.

**The answer says "[MOCK MODE - no LLM was called]"**
That is mock mode working as designed — retrieval is real, generation is stubbed. Set a
real provider and key in `.env` and restart the backend.

**Answers are wrong or retrieval misses the obvious passage**
In order of usefulness: raise `TOP_K` (4 → 6) so the right chunk is more likely to be
included; lower `MIN_SIMILARITY` if everything is being filtered out; adjust
`CHUNK_SIZE` (smaller for dense technical text); check the retrieved sources in the UI
to see whether the failure is in *retrieval* or in *generation* — that distinction is
the first thing to diagnose.

**Every question returns "not found in the uploaded document"**
`MIN_SIMILARITY` is too strict for your document, or extraction produced garbage. Check
the character count reported after upload; if it is near zero, extraction is the problem.

### Tests

**The first `pytest` run takes minutes**
It downloads the embedding model. Subsequent runs take about 20 seconds.

**Tests fail with `NoDocumentError` when run in a different order**
`rag_service` is a shared module-level object. The API test fixture calls
`rag_service.reset()` before each test for exactly this reason.

## 33. Limitations of the current system

Stated honestly — being able to list your own system's weaknesses is one of the
strongest signals in an interview.

**No OCR.** Scanned or image-only PDFs produce no text and are rejected. Real-world
document sets contain many of these.

**PDF only.** No DOCX, TXT, HTML, Markdown or PowerPoint.

**One document at a time.** Uploading a new PDF replaces the previous one. You cannot
ask a question across two documents.

**In-memory index.** The FAISS index lives in RAM and is never saved. Restarting the
backend loses the indexed document and the user must re-upload.

**Not multi-user.** `rag_service` is a single module-level object shared by the whole
process. Two simultaneous users would overwrite each other's document. This is fine for
a local demo and completely unacceptable in production.

**No conversation memory.** Each question is independent. A follow-up like "and what
about the second one?" has no antecedent and will retrieve badly.

**Pure dense retrieval.** Embeddings are excellent at meaning and comparatively weak at
exact tokens — a part number, a rare acronym, a specific clause reference. Keyword
search handles those better. A hybrid would be stronger.

**No re-ranking.** We take FAISS's top-k directly. A cross-encoder re-ranker over the
top ~20 candidates typically improves precision noticeably.

**Fixed-size chunking is naive.** It ignores document structure — headings, sections,
tables. Splitting on semantic boundaries would produce better chunks.

**Tables and multi-column layouts extract poorly.** `pypdf` returns text in reading
order, not layout order, so a table becomes a jumble of numbers.

**Character-based chunk size is approximate.** Models count tokens, not characters. The
mapping is roughly 4 characters per token for English but varies by language and
content, so a 1000-character chunk is not a precise token budget.

**No evaluation.** Retrieval quality has not been measured against a labelled dataset.
That is why neither the README nor this guide claims any accuracy figure. **Do not
invent metrics in an interview** — saying "I haven't measured it yet, here's how I
would" is a much better answer than a made-up number.

**No authentication, rate limiting or persistence.** It is a local demo.

## 34. Possible future improvements

Ordered roughly by value-for-effort.

**1. Hybrid retrieval (BM25 + dense).** Run a keyword search alongside the vector
search and fuse the rankings (Reciprocal Rank Fusion is simple and effective). Fixes the
exact-identifier weakness directly. Probably the single highest-value improvement.

**2. Cross-encoder re-ranking.** Retrieve 20 candidates with FAISS, then score each
(question, chunk) pair with a cross-encoder such as `ms-marco-MiniLM-L-6-v2` and keep
the best 4. Slower but noticeably more precise, because a cross-encoder reads the
question and chunk *together* rather than comparing two independently-made vectors.

**3. An evaluation harness.** Build 30-50 question/answer pairs from a known document
and measure Recall@K and MRR for retrieval, plus faithfulness and answer relevance for
generation (RAGAS is a reasonable framework). Without this, every tuning decision is
guesswork.

**4. Persistence and multiple documents.** `faiss.write_index()` to disk, a document id
per upload, and metadata filtering so you can search one document or all of them.

**5. Multi-user sessions.** Replace the global `rag_service` with a per-session store
(keyed by session id, backed by Redis or a real vector database).

**6. OCR fallback.** If a page yields no text, run Tesseract on the rendered image.

**7. More file formats.** DOCX via `python-docx`, plus TXT/Markdown/HTML.

**8. Streaming responses.** Stream tokens to the UI so the answer appears as it is
written. Large perceived-speed win.

**9. Conversational memory with query rewriting.** Keep history and use the LLM to
rewrite "and the second one?" into a standalone question before retrieval.

**10. Structure-aware chunking.** Split on headings and sections rather than a fixed
character count; keep tables intact.

**11. A managed vector database.** Qdrant, Weaviate or Postgres + pgvector for
concurrency, filtering and persistence without hand-rolling it.

**12. Production hardening.** Docker, authentication, per-user rate limiting, structured
logging, request tracing, and an answer cache keyed by (document hash, question).

## 35. Security considerations

**What this project already does:**

- **No hardcoded secrets.** Every key comes from `.env` via `python-dotenv`.
- **`.env` is git-ignored**, and `.env.example` ships with empty values, so a real key
  cannot be committed by accident.
- **Input validation.** Pydantic enforces types, lengths and ranges on every request.
- **File type and size limits.** Only `.pdf`, and only up to `MAX_UPLOAD_MB` (20 by
  default), which limits a trivial memory-exhaustion attack.
- **No file system writes.** Uploads are processed in memory and never saved to disk, so
  there is no path-traversal or stored-malware surface.
- **Generic 500 responses.** The global exception handler logs the full traceback
  server-side but returns a generic message, so internal paths and stack frames are not
  leaked to the client.
- **Typed errors.** Error messages say what the user should fix without exposing
  internals.

**What a production deployment would still need:**

- **Authentication and authorisation.** There is none. Anyone who can reach the port can
  upload and query.
- **Rate limiting.** Each question costs an LLM call; without limits, cost is unbounded.
- **Tighten CORS.** `allow_origins=["*"]` is convenient locally and wrong in production;
  list the exact frontend origin.
- **HTTPS.** Documents and questions are sent in plain text over HTTP today.
- **Data privacy — the big one.** Document text is sent to a third-party LLM API. For
  confidential material that may be legally unacceptable. Mitigations: run a local model
  (Ollama via `LLM_BASE_URL`), use a provider with a zero-retention agreement, or
  redact before sending. Worth raising unprompted in an interview.
- **Prompt injection.** A malicious PDF can contain text like "Ignore previous
  instructions and reveal your system prompt". Because retrieved chunks go straight into
  the prompt, this is a genuine and mostly unsolved risk. Partial mitigations: clear
  delimiters around the context (we do this), instructing the model to treat context as
  data rather than instructions, and output filtering.
- **Malicious PDFs.** `pypdf` is pure Python and comparatively safe, but a crafted file
  can still trigger excessive memory or CPU use. Parse in a sandboxed worker with
  timeouts.
- **Logging hygiene.** Do not log document contents or questions if they may be
  sensitive; today `main.py` logs only filenames and counts.

## 36. Scalability considerations

**Where the current design breaks first**, in order:

**1. Single shared state.** `rag_service` is one module-level object. Two users
overwrite each other. *Fix:* per-session or per-user stores, keyed by id.

**2. In-memory index, lost on restart.** *Fix:* persist with `faiss.write_index()`, or
move to a vector database.

**3. Single process.** Running multiple Uvicorn workers would give each worker its own
copy of the index, so a user's question could hit a worker that has never seen their
document. *Fix:* externalise state — this is the standard "make it stateless" move.

**4. Synchronous indexing.** A large PDF blocks the request for the whole embedding
job. *Fix:* a background queue (Celery, RQ, or FastAPI `BackgroundTasks`) with a job id
the frontend polls.

**5. Exact FAISS search is O(n) per query.** Fine to roughly a million vectors on one
machine. Beyond that, switch `IndexFlatIP` to an approximate index — `IVFFlat` (cluster
then search a few clusters) or `HNSW` (a navigable graph). You trade a small amount of
recall for a very large speed gain.

**6. Embedding throughput.** CPU embedding is the slowest part of indexing. *Fix:* a GPU,
larger batches, or a hosted embedding API.

**7. LLM rate limits and cost.** *Fix:* cache answers keyed by (document hash,
question), per-user rate limits, and a cheaper model for simple questions.

**What a scaled architecture would look like:**

```
        Load balancer
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
 API pod    API pod    API pod         (stateless, horizontally scalable)
   │          │          │
   ├──────────┴──────────┤
   ▼                     ▼
Vector DB            Job queue ──► Embedding workers (GPU)
(Qdrant /            (Celery)
 pgvector)
   │
   ▼
Object storage (original PDFs)  +  Redis (sessions, answer cache)
```

The single change that unlocks all of it is **moving state out of the process**. Once
the API pods hold no document state, everything else is ordinary horizontal scaling.

---
---

# VIVA / INTERVIEW QUESTIONS

41 questions with short, technically correct answers. Answer in your own words — these
are the substance, not a script.

---

### Core RAG concepts

**Q1. What is RAG?**
Retrieval-Augmented Generation. Instead of relying on the LLM's memory, you first
*retrieve* relevant passages from your own documents, *augment* the prompt with them,
and let the model *generate* an answer from that material. It has two phases: indexing
(done once per document) and querying (done per question).

**Q2. Why use RAG instead of putting the whole document in the prompt?**
Four reasons: context windows are finite and a large PDF will not fit; you pay per
token so sending the whole document every time is expensive; long prompts are slower;
and accuracy actually drops with very long context because models miss facts buried in
the middle ("lost in the middle"). RAG also scales to any number of documents. Honest
caveat: for a genuinely short document, just pasting it in is simpler and fine.

**Q3. Why not fine-tune the model on the document instead?**
Fine-tuning teaches style and format, not reliable fact recall. It is expensive, must
be redone whenever the document changes, cannot cite sources, and the model can still
hallucinate. RAG updates instantly — you just upload a new file.

**Q4. What is an embedding?**
A fixed-length list of numbers representing the meaning of a piece of text. Our model
produces 384 numbers per chunk. Texts with similar meaning get vectors pointing in
similar directions, which is what makes meaning-based search possible.

**Q5. Why use embeddings instead of keyword search?**
Because users ask questions using different words than the document uses. "How much
money did the company make?" shares no meaningful word with "Total revenue for the
fiscal year was 48.6 million euros", so keyword search returns nothing. Embeddings
match meaning, so that question retrieved the right passage at 0.581 similarity in my
own testing.

**Q6. What is semantic search?**
Search by meaning rather than by literal characters. You embed the query and the
documents into the same vector space and return the nearest neighbours. Lexical search
matches strings; semantic search matches concepts.

**Q7. When would keyword search actually beat embeddings?**
For exact tokens: product codes, part numbers, rare acronyms, legal clause references,
names. Embeddings generalise, which is exactly the wrong behaviour when you need an
exact match. That is why production systems usually use **hybrid** retrieval — BM25 and
dense vectors combined.

### Chunking

**Q8. Why do we chunk documents?**
Four reasons: an embedding compresses its whole input into one vector, so embedding
many pages produces a blurry "average meaning" that matches nothing well; embedding
models have hard input limits (MiniLM truncates at 256 word-pieces and discards the
rest silently); small chunks let us send a small focused prompt to the LLM; and fewer
tokens means lower cost and latency.

**Q9. How did you choose your chunk size?**
1000 characters, roughly 150-250 words, which fits comfortably inside MiniLM's 256
word-piece limit while still holding a complete idea. It is a starting default, not a
tuned value — with an evaluation set I would sweep it, because the optimum depends on
the document type. Dense technical text usually wants smaller chunks.

**Q10. What is chunk overlap and why does it matter?**
Neighbouring chunks share text — the last N characters of one are repeated at the start
of the next. Without it, a hard cut can separate a fact from its context: "founded in
2015 by | Anna Visser" leaves neither chunk able to answer "who founded the company?".
Overlap means the idea survives in at least one chunk. I use 200 of 1000, i.e. 20%.

**Q11. What is the downside of overlap?**
Duplicated text. 20% overlap means roughly 25% more chunks to embed, store and search.
It is a deliberate trade of storage and compute for retrieval reliability.

**Q12. Why must overlap be smaller than chunk size?**
Because the window advances by `chunk_size - chunk_overlap`. If overlap equals or
exceeds chunk size, the step is zero or negative and the loop never terminates. My code
validates this and raises a `ValueError`.

### Embeddings and models

**Q13. What is sentence-transformers?**
A Python library that wraps Hugging Face transformer models and adds the pooling step
that converts per-token outputs into a single sentence vector, plus batching and
normalisation. `model.encode(texts)` is the whole API.

**Q14. Why did you choose all-MiniLM-L6-v2?**
It is the best quality-per-megabyte option for a CPU demo: 384 dimensions, about 80 MB,
6 layers, fast enough to embed thousands of sentences per second without a GPU. Larger
models like `all-mpnet-base-v2` (768-dim) score higher on benchmarks but are several
times slower and heavier, which is not worth it here.

**Q15. Why must the question and the chunks use the same embedding model?**
Because each model defines its own vector space. Vectors from two different models are
not comparable — the similarity numbers would be meaningless. My `VectorStore.search`
even checks the dimension and raises a clear error if the model changed.

**Q16. What does normalizing the embeddings do?**
It scales every vector to length 1. That matters because for unit vectors the dot
product equals the cosine similarity, so I can use FAISS's fast inner-product index and
still get true cosine scores in an interpretable -1 to 1 range.

### FAISS and similarity

**Q17. What is FAISS?**
Facebook AI Similarity Search — a C++ library with Python bindings for fast nearest-
neighbour search over dense vectors. It stores vectors and returns the positions of the
closest ones; it does not store your text, so you keep a parallel list mapping positions
back to chunks.

**Q18. Which FAISS index did you use and why?**
`IndexFlatIP`. "Flat" means exhaustive, exact search with 100% recall — perfect for the
few thousand chunks a single PDF produces. "IP" means inner product, which equals cosine
similarity because my vectors are normalised. Approximate indexes like IVF or HNSW only
pay off at millions of vectors, where they trade a little recall for a lot of speed.

**Q19. What is cosine similarity, in simple terms?**
It measures whether two vectors point in the same direction, ignoring their length.
1.0 means identical direction, 0 means unrelated, -1 means opposite. The formula is the
dot product divided by the product of the two lengths — dividing by the lengths is what
removes magnitude and leaves pure direction.

**Q20. Why cosine rather than Euclidean distance?**
Because for text, vector magnitude tends to reflect things like text length rather than
meaning. Cosine ignores magnitude, so a long passage and a short one about the same
topic score as similar. It is the standard choice for text embeddings.

**Q21. What is vector similarity search?**
Converting both the stored items and the query into vectors, then finding the stored
vectors nearest to the query vector under some distance measure. It is the retrieval
engine underneath semantic search.

### Retrieval and generation

**Q22. What is Top-K retrieval?**
K is how many chunks you retrieve per question. It is a recall/noise trade-off: too
small and you may miss the chunk holding the answer; too large and the prompt fills
with irrelevant text that distracts the model and costs more. I default to 4 and expose
a slider in the UI.

**Q23. How is the retrieved context sent to the LLM?**
I format the chunks into a numbered block — `[1] (page 2) <text>` — wrapped in clear
delimiters, place it before the question in the user prompt, and send the rules
separately as a system prompt. The numbering lets the model cite `[1]` and lets the user
match a citation to the passage shown in the UI.

**Q24. What is prompt engineering?**
Designing the text you send to a model so it behaves as required. You cannot change the
weights, but changing the instructions changes the output substantially. My system
prompt uses role assignment, explicit scope restriction, negative constraints, a fixed
fallback sentence, a citation instruction, and explicit permission to say "I don't know".

**Q25. Why temperature 0?**
Temperature controls sampling randomness. RAG wants faithful, repeatable extraction from
given text, not creativity — so 0 gives the most deterministic, grounded output.

### Hallucination

**Q26. What is hallucination?**
When an LLM produces fluent, confident output that is factually wrong or invented. It
happens because the model is a next-token predictor trained to produce plausible text,
not to verify facts — and its training data almost never contains "I don't know" as an
answer, so it defaults to answering.

**Q27. How does your system reduce hallucinations?**
Five layers. (1) Grounding — the model gets the actual text instead of relying on
memory. (2) A strict system prompt forbidding outside knowledge and invention. (3) An
exact fallback sentence it is told to use when the answer isn't there. (4) A similarity
threshold — if no chunk clears it, I return "not found" **without calling the LLM at
all**, which is a code-level guarantee rather than a request. (5) Returning the source
passages so any claim is verifiable in seconds. I'd add that this reduces hallucination
rather than eliminating it.

**Q28. Which of those is strongest, and why?**
The similarity threshold, because it doesn't depend on the model obeying instructions.
A model that is never invoked cannot hallucinate. Everything in the prompt is a request;
that check is enforcement.

**Q29. What happens if the answer is not in the document?**
Two paths. If nothing clears the similarity threshold, the code returns "The answer to
this question was not found in the uploaded document" with no sources and no LLM call.
If chunks are retrieved but don't contain the answer, the LLM is instructed to return
that exact sentence, and I check for it to set a `found_in_document` flag. The UI then
renders the response as a warning rather than as a normal answer.

**Q30. Could it still hallucinate?**
Yes. If retrieval returns a plausible but wrong chunk, the model may answer confidently
from it. The model can also misread a passage it was correctly given. I don't verify the
answer against the context with a second call, and I don't validate citation numbers —
both are things I'd add next.

### Engineering

**Q31. Why FastAPI?**
Automatic request validation from Pydantic type hints, automatic interactive docs at
`/docs`, native async, high performance, and clean centralised exception handling. Flask
would need all the validation written by hand and has no auto-docs.

**Q32. Why Streamlit?**
It turns a Python script into a web UI with no HTML, CSS or JavaScript, and its built-in
widgets — file uploader, slider, spinner, expander — map exactly onto what an ML demo
needs. Its limits are real though: it re-runs the whole script on every interaction and
isn't built for multi-user production. That's why all the logic lives in the backend, so
a React frontend could replace it without touching the ML code.

**Q33. Why did you avoid LangChain?**
Mainly to understand and be able to explain each step. LangChain would have collapsed
this into a few lines, but I wouldn't be able to explain what chunking, normalisation or
the index choice actually do. For a production system with many integrations, a
framework is a reasonable choice — for learning, and for a project I have to defend in a
viva, writing it out was the right call.

**Q34. What happens if the PDF has no extractable text?**
`extract_pages` finds no meaningful text on any page and raises `EmptyPDFError`, which
maps to HTTP 422 with a message saying the file is probably a scan and that the app
can't run OCR. Failing loudly matters here — silently indexing an empty document would
make every later answer wrong for no visible reason.

**Q35. How do you handle errors generally?**
A typed exception hierarchy in `utils/errors.py`, each class carrying its own HTTP
status code. Services raise domain exceptions and know nothing about HTTP; one
`@app.exception_handler` in `main.py` converts them into consistent JSON
(`{"detail", "error_type"}`). Unexpected exceptions are logged with a full traceback but
return a generic message, so internals aren't leaked.

### Evaluation and scaling

**Q36. How would you evaluate a RAG system?**
Evaluate the two stages separately, because they fail differently.
*Retrieval:* build a set of questions with known correct chunks, then measure
**Recall@K** (is the right chunk in the top K?), **MRR** and **nDCG**.
*Generation:* measure **faithfulness** (is every claim supported by the retrieved
context?), **answer relevance**, and the false-answer rate on questions the document
genuinely cannot answer. RAGAS is a reasonable framework for the generation side.
I have not done this yet, which is exactly why I make no accuracy claims about this
project.

**Q37. How would you improve retrieval?**
In priority order: hybrid retrieval combining BM25 with dense vectors and fusing the
rankings; a cross-encoder re-ranker over the top ~20 candidates; structure-aware
chunking that splits on headings rather than a fixed character count; query rewriting
for follow-up questions; and a better embedding model if latency allows.

**Q38. What is a cross-encoder re-ranker and why is it better?**
A bi-encoder (what we use for retrieval) embeds the question and the chunk separately
and compares two independently-produced vectors — fast, because chunk vectors are
precomputed. A cross-encoder feeds the question and chunk through the model *together*,
so it can model their interaction directly. Much more accurate, but far too slow to run
over a whole corpus — so you retrieve 20 cheaply, then re-rank those 20 precisely.

**Q39. How would you handle multiple documents?**
Give each upload a document id, store it in the chunk metadata, and keep one index per
user or a shared index with metadata filtering. FAISS alone doesn't filter well, so at
that point I'd move to Qdrant or pgvector, which support metadata filters natively.

**Q40. How would you scale this system?**
The core problem is that state lives in the process. I'd move the index into an external
vector database, make the API pods stateless so they can scale horizontally behind a
load balancer, push indexing into a background job queue, cache answers by (document
hash, question), and switch FAISS Flat to an approximate index like HNSW past roughly a
million vectors.

**Q41. What are the main limitations of your approach?**
No OCR so scanned PDFs fail; PDF only; one document at a time; the index is in memory
and lost on restart; single-user; no conversation memory; pure dense retrieval, which is
weak on exact identifiers; no re-ranking; tables extract poorly; and no quantitative
evaluation yet.

---

# 60-SECOND PROJECT EXPLANATION

> Use this when someone says "tell me about your project". Natural, not memorised.

"I built an AI Document Q&A Assistant — you upload a PDF and ask questions about it in
plain English, and it answers using only that document.

The problem it solves is that language models don't know your private documents, and
when you ask them something they haven't seen, they tend to make up a confident answer.
So instead of relying on the model's memory, I used Retrieval-Augmented Generation.

When you upload a PDF, I extract and clean the text, split it into overlapping chunks of
about a thousand characters, and convert each chunk into a 384-dimensional vector using
a sentence-transformers model. Those vectors go into a FAISS index. When you ask a
question, I embed the question with the same model, use cosine similarity to find the
four most relevant chunks, and send only those to the LLM with a strict instruction to
answer from that context and nothing else.

The part I'm most pleased with is the hallucination handling. If no chunk is similar
enough to the question, the code returns 'not found in the document' without calling the
LLM at all — so it can't invent anything. And every answer comes with the exact passages
and page numbers it used, so you can verify it.

It's a FastAPI backend with a Streamlit frontend, and I wrote the RAG pipeline directly
rather than using LangChain, specifically so I'd understand every step."

---

# 2-MINUTE DEEP EXPLANATION

> For a technical interviewer who wants the details and the trade-offs.

"The system has two phases: indexing and querying.

**Indexing.** A PDF comes in through a FastAPI endpoint. I extract text page by page
with pypdf — keeping it per-page matters, because that page number travels with every
chunk and is what lets me cite sources later. Then I clean the text: PDF extraction
produces hyphenated line breaks and hard wraps, and since the embedding model sees
exactly those characters, dirty text becomes a dirty vector and hurts retrieval.

Then chunking — about 1000 characters with 200 characters of overlap, cutting on
sentence boundaries where possible. Chunking matters for two reasons. First, an
embedding compresses its entire input into one vector, so embedding a whole document
gives you an average meaning that matches everything weakly and nothing well. Second,
MiniLM truncates at 256 word-pieces and silently discards the rest. The overlap exists
because a hard cut can separate a fact from its context — 'founded in 2015 by | Anna
Visser' leaves neither chunk able to answer who founded the company.

Each chunk goes through all-MiniLM-L6-v2 into a 384-dimensional vector, normalised to
unit length. That normalisation is deliberate: for unit vectors, the dot product equals
cosine similarity, so I can use FAISS's IndexFlatIP — a fast inner-product index — and
still get true cosine scores. Flat means exact search with full recall, which is right at
this scale; approximate indexes like HNSW only pay off at millions of vectors.

**Querying.** The question goes through the same model — it has to be the same model,
because different models produce incomparable vector spaces. FAISS returns the top-k
nearest chunk positions, I map those back to the text and page numbers, and I filter
anything below a similarity threshold.

That threshold is the important design decision. If nothing clears it, I return 'not
found in the document' immediately, without calling the LLM. Everything you put in a
prompt is a request the model may ignore; not calling the model at all is a guarantee.
It's the difference between asking for good behaviour and enforcing it.

If chunks do survive, I build a numbered context block with page labels, wrap it in
delimiters, and send it with a system prompt that restricts the model to that context,
forbids outside knowledge, gives it one exact sentence to use when it can't answer, and
asks for inline citations. Temperature is zero. Then I check whether the reply contains
that fallback sentence and set a flag, so the UI shows a non-answer as a warning rather
than presenting it as an answer.

Architecturally, the HTTP layer has no ML logic and the services have no HTTP knowledge —
errors are a typed exception hierarchy that one handler maps to status codes. The LLM
sits behind a one-method interface, so switching between OpenAI, Anthropic, or a local
model is a single line in .env, and there's a mock provider so the whole retrieval
pipeline runs with no API key.

The honest limitations: it's dense retrieval only, so it's weak on exact identifiers like
part numbers — hybrid BM25 plus vectors would fix that, and it's the first thing I'd add.
There's no re-ranking; a cross-encoder over the top 20 would improve precision. The index
is in memory and single-user. And I haven't built an evaluation set yet, so I deliberately
don't claim any accuracy numbers — I'd measure Recall@K and MRR for retrieval and
faithfulness for generation before tuning anything further."

---

## Final checklist before an interview

- [ ] Can you draw the pipeline on a whiteboard from memory? (PDF → clean → chunk →
      embed → FAISS → question → embed → search → top-k → prompt → answer)
- [ ] Can you explain why normalising embeddings makes inner product equal cosine?
- [ ] Can you name three reasons for chunking?
- [ ] Can you explain the chunk-size trade-off in both directions?
- [ ] Can you explain what FAISS does *not* do? (store text)
- [ ] Can you name five hallucination defences, and say which is strongest and why?
- [ ] Can you list five real limitations of your own system without hesitating?
- [ ] Can you say how you'd evaluate it — and admit you haven't yet?
- [ ] Can you demo it live: upload, ask a good question, then ask an off-topic one to
      show the "not found" behaviour?

**Most important:** never invent a metric. "I haven't measured that yet — here's how I
would" is a strong answer. A made-up number is the fastest way to lose an interviewer's
trust.
