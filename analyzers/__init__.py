# ragtrace/analyzers/__init__.py
from __future__ import annotations
from ragtrace.session import TraceSession
from ragtrace.config import TracerConfig
from ragtrace.analyzers.retrieval import analyze_retrieval
from ragtrace.analyzers.context import analyze_context
from ragtrace.analyzers.generation import analyze_generation

def run_all_analyzers(session: TraceSession, config: TracerConfig) -> None:


    for span in session.retrieval_spans:
        analyze_retrieval(span, config=config)

    # context analysis needs both retrieval and generation spans
    if session.has_retrieval and session.has_generation:
        analyze_context(
            retrieval_span=session.retrieval_spans[-1],
            generation_span=session.generation_spans[-1],
            config=config,
        )

    for span in session.generation_spans:
        analyze_generation(span, config=config)
