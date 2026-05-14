from __future__ import annotations
from collections.abc import Sequence
from typing import Any
from ragtrace.session import RetrievalSpan, GenerationSpan
from ragtrace.config import TracerConfig

_model = None


def _get_embedding_model():
    global _model
    if _model is None:
        from rich.console import Console
        from sentence_transformers import SentenceTransformer

        Console().print("[dim]Loading embedding model (first run)...[/dim]")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _cosine_similarity(a, b) -> float:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    import numpy as np

    score = float(sk_cosine([a], [b])[0][0])
    return max(0.0, min(1.0, score))


def _record_note(
    span,
    *,
    code: str,
    severity: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    note = {
        "code": code,
        "severity": severity,
        "message": message,
        "details": details,
    }
    span.analysis_notes.append(note)
    span.diagnosis.append(message)
    return note


def analyze_context(
    retrieval_spans: RetrievalSpan | Sequence[RetrievalSpan],
    generation_span: GenerationSpan,
    config: TracerConfig,
) -> dict[str, Any]:
    """
    Checks how well the generated response used the retrieved context.
    Requires semantic=True in config (uses sentence-transformers).
    """
    if isinstance(retrieval_spans, RetrievalSpan):
        retrieval_spans = [retrieval_spans]

    report: dict[str, Any] = {
        "generation_event_index": generation_span.event_index,
        "retrieval_event_indices": [span.event_index for span in retrieval_spans],
        "multi_retrieval": len(retrieval_spans) > 1,
        "pairings": [],
        "findings": [],
    }

    if len(retrieval_spans) > 1:
        note = _record_note(
            generation_span,
            code="multi_retrieval",
            severity="info",
            message=(
                f"Multi-hop retrieval signal: {len(retrieval_spans)} retrieval steps "
                f"were linked to this answer."
            ),
            retrieval_count=len(retrieval_spans),
            retrieval_event_indices=report["retrieval_event_indices"],
        )
        report["findings"].append(note)

    if not config.semantic:
        return report

    model = _get_embedding_model()
    response = generation_span.response
    if not response:
        return report

    context_word_count = 0
    best_overall: dict[str, Any] | None = None

    for retrieval_span in retrieval_spans:
        chunks = retrieval_span.chunks
        pair_report: dict[str, Any] = {
            "retrieval_event_index": retrieval_span.event_index,
            "chunk_count": len(chunks),
            "findings": [],
        }

        if not chunks:
            report["pairings"].append(pair_report)
            continue

        all_texts = [response] + chunks
        embeddings = model.encode(all_texts, normalize_embeddings=True)
        response_emb = embeddings[0]
        chunk_embs = embeddings[1:]

        similarities = [
            _cosine_similarity(response_emb, chunk_emb) for chunk_emb in chunk_embs
        ]

        best_chunk_idx = similarities.index(max(similarities))
        best_score = similarities[best_chunk_idx]
        context_word_count += sum(len(chunk.split()) for chunk in chunks)

        pair_report["best_chunk_index"] = best_chunk_idx
        pair_report["best_similarity"] = round(best_score, 4)

        if best_overall is None or best_score > best_overall["best_similarity"]:
            best_overall = {
                "retrieval_span": retrieval_span,
                "best_chunk_index": best_chunk_idx,
                "best_similarity": best_score,
            }

        if best_chunk_idx > config.lost_in_middle_position:
            original_score = retrieval_span.scores[best_chunk_idx]
            top_score = retrieval_span.scores[0]
            message = (
                f"Lost-in-the-middle signal: response is most similar to chunk "
                f"#{best_chunk_idx + 1} (retrieval score: {original_score:.2f}), "
                f"not chunk #1 (score: {top_score:.2f}). "
                f"Consider reranking chunks by response-relevance before generation."
            )
            note = _record_note(
                generation_span,
                code="lost_in_the_middle",
                severity="warn",
                message=message,
                retrieval_event_index=retrieval_span.event_index,
                retrieval_chunk_index=best_chunk_idx,
                response_similarity=round(best_score, 4),
                retrieval_score=round(original_score, 4),
                top_score=round(top_score, 4),
            )
            retrieval_span.analysis_notes.append(note)
            retrieval_span.diagnosis.append(message)
            pair_report["findings"].append(note)
            report["findings"].append(note)

        report["pairings"].append(pair_report)

    if best_overall is not None and best_overall["best_similarity"] < 0.4:
        note = _record_note(
            generation_span,
            code="low_context_utilisation",
            severity="warn",
            message=(
                f"Low context utilisation — response has low semantic similarity "
                f"to the retrieved chunks (best: {best_overall['best_similarity']:.2f}). "
                f"The LLM may be answering from its training weights rather than "
                f"the provided context."
            ),
            best_similarity=round(best_overall["best_similarity"], 4),
            retrieval_event_index=best_overall["retrieval_span"].event_index,
            retrieval_chunk_index=best_overall["best_chunk_index"],
        )
        report["findings"].append(note)

    response_word_count = len(response.split())
    if response_word_count < 20 and context_word_count > 200:
        note = _record_note(
            generation_span,
            code="short_response_over_context",
            severity="warn",
            message=(
                f"Response ({response_word_count} words) is very short relative "
                f"to context ({context_word_count} words). "
                f"Consider whether the model has enough information or if the "
                f"prompt is over-constraining response length."
            ),
            response_word_count=response_word_count,
            context_word_count=context_word_count,
        )
        report["findings"].append(note)

    return report
