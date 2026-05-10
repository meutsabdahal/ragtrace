import pytest
from ragtrace.session import RetrievalSpan, GenerationSpan
from ragtrace.config import TracerConfig
from ragtrace.analyzers.retrieval import analyze_retrieval
from ragtrace.analyzers.generation import analyze_generation


def make_retrieval_span(scores):
    return RetrievalSpan(
        query="test",
        chunks=[f"chunk {i}" for i in range(len(scores))],
        scores=scores,
        k_requested=len(scores),
        k_returned=len(scores),
    )


def test_retrieval_low_scores_flagged():
    span = make_retrieval_span([0.3, 0.3, 0.3, 0.3, 0.3])
    analyze_retrieval(span, config=TracerConfig())
    assert any("low-relevance" in d for d in span.diagnosis)


def test_retrieval_healthy_scores_pass():
    span = make_retrieval_span([0.85, 0.82, 0.80, 0.78, 0.75])
    analyze_retrieval(span, config=TracerConfig())
    assert any("healthy" in d for d in span.diagnosis)


def test_retrieval_large_gap_flagged():
    span = make_retrieval_span([0.92, 0.40, 0.38, 0.35, 0.33])
    analyze_retrieval(span, config=TracerConfig())
    assert any("gap" in d for d in span.diagnosis)


def test_retrieval_k_mismatch_flagged():
    span = RetrievalSpan(
        query="test",
        chunks=["a", "b"],
        scores=[0.8, 0.7],
        k_requested=5,
        k_returned=2,
    )
    analyze_retrieval(span, config=TracerConfig())
    assert any("returned" in d for d in span.diagnosis)


def test_generation_hedging_flagged():
    span = GenerationSpan(
        prompt="prompt",
        response="I believe this is generally true and typically works this way.",
        model="test",
    )
    analyze_generation(span, config=TracerConfig())
    assert any("hedging" in d.lower() for d in span.diagnosis)


def test_generation_healthy_response():
    span = GenerationSpan(
        prompt="prompt",
        response="The NEPSE index reached 3,198 points on August 12, 2023.",
        model="test",
    )
    analyze_generation(span, config=TracerConfig())
    assert any("healthy" in d for d in span.diagnosis)
