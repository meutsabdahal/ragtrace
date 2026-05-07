# ragtrace
**A lightweight debugger for RAG pipelines.**

When a RAG pipeline gives a wrong answer, you have no idea why. Was it retrieval? Did
the right documents come back but get ranked wrong? Did the LLM ignore the context
entirely? `ragtrace` wraps your existing pipeline with one decorator and shows you
exactly where it broke.

---

## Why this exists

Most RAG debugging looks like this: print retrieved chunks to stdout, read them
manually, guess what went wrong. That's not debugging that's hoping.

`ragtrace` gives you a structured trace of every query: what was retrieved, the
similarity score for each chunk, the exact prompt sent to the model, and a diagnosis
of where the pipeline is weak. Framework-agnostic works with ChromaDB, FAISS,
Qdrant, anything.

---

## How it works

1. `@trace` wraps your function and creates a `TraceSession`
2. Session ID is stored in a `contextvars.ContextVar` — propagates correctly
   through both sync and async code
3. `log_retrieval()` and `log_generation()` read the `ContextVar` and append
   spans to the active session
4. After your function returns, three analyzers run:
   - **Retrieval analyzer**: score distributions, low-relevance padding
   - **Context analyzer**: chunk-response similarity, lost-in-middle detection
   - **Generation analyzer**: hedging language, entity mismatch signals
5. Terminal renderer prints the trace; HTML renderer saves a shareable report

The embedding model runs entirely locally your data never leaves your machine.