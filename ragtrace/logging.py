from __future__ import annotations
import time
from typing import Optional
from ragtrace.collector import get_collector
from ragtrace.session import RetrievalSpan, GenerationSpan


def log_retrieval(
    query: str,
    chunks: list[str],
    scores: list[float],
    k_requested: Optional[int] = None,
) -> None:
    """
    Call this immediately after your retrieval step.

    Args:
        query:       the query string sent to the retriever
        chunks:      list of retrieved text chunks
        scores:      similarity scores, one per chunk (higher = more relevant)
        k_requested: how many results you asked for (defaults to len(chunks))
    """
    from typing import Optional  # local import to avoid circular

    collector = get_collector()
    session = collector.get_current_session()

    if session is None:
        # called outside a @trace context — silently no-op
        # this lets users leave log_retrieval in production code
        # without crashing if tracing is disabled
        return

    span = RetrievalSpan(
        query=query,
        chunks=chunks,
        scores=scores,
        k_requested=k_requested if k_requested is not None else len(chunks),
        k_returned=len(chunks),
    )
    session.add_retrieval_span(span)


def log_generation(
    prompt: str,
    response: str,
    model: str,
    prompt_tokens: int = 0,
    response_tokens: int = 0,
) -> None:
    """
    Call this immediately after your generation step.

    Args:
        prompt:          the full prompt sent to the LLM
        response:        the LLM's response text
        model:           model name string (e.g. "llama3.2")
        prompt_tokens:   token count (estimated from chars if 0)
        response_tokens: token count (estimated from chars if 0)
    """
    collector = get_collector()
    session = collector.get_current_session()

    if session is None:
        return

    span = GenerationSpan(
        prompt=prompt,
        response=response,
        model=model,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
    )
    session.add_generation_span(span)
