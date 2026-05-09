from __future__ import annotations
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


def analyze_context(
    retrieval_span: RetrievalSpan,
    generation_span: GenerationSpan,
    config: TracerConfig,
) -> None:
    """
    Checks how well the generated response used the retrieved context.
    Requires semantic=True in config (uses sentence-transformers).
    """
    if not config.semantic:
        return

    model = _get_embedding_model()
    response = generation_span.response
    chunks = retrieval_span.chunks

    if not chunks or not response:
        return

    # embed the response and all chunks
    all_texts = [response] + chunks
    embeddings = model.encode(all_texts, normalize_embeddings=True)
    response_emb = embeddings[0]
    chunk_embs = embeddings[1:]

    # compute similarity between response and each chunk
    similarities = [
        _cosine_similarity(response_emb, chunk_emb) for chunk_emb in chunk_embs
    ]

    best_chunk_idx = similarities.index(max(similarities))
    best_score = similarities[best_chunk_idx]

    # Heuristic 1: lost in the middle
    # the most relevant chunk (by response similarity) is not at position 0
    if best_chunk_idx > config.lost_in_middle_position:
        original_score = retrieval_span.scores[best_chunk_idx]
        top_score = retrieval_span.scores[0]
        generation_span.diagnosis.append(
            f"Lost-in-the-middle signal: response is most similar to chunk "
            f"#{best_chunk_idx + 1} (retrieval score: {original_score:.2f}), "
            f"not chunk #1 (score: {top_score:.2f}). "
            f"Consider reranking chunks by response-relevance before generation."
        )

    # Heuristic 2: response barely uses the context at all
    if best_score < 0.4:
        generation_span.diagnosis.append(
            f"Low context utilisation — response has low semantic similarity "
            f"to all retrieved chunks (best: {best_score:.2f}). "
            f"The LLM may be answering from its training weights rather than "
            f"the provided context."
        )

    # Heuristic 3: response is very short relative to context
    context_word_count = sum(len(c.split()) for c in chunks)
    response_word_count = len(response.split())
    if response_word_count < 20 and context_word_count > 200:
        generation_span.diagnosis.append(
            f"Response ({response_word_count} words) is very short relative "
            f"to context ({context_word_count} words). "
            f"Consider whether the model has enough information or if the "
            f"prompt is over-constraining response length."
        )
