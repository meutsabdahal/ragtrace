import pytest
from ragtrace.session import RetrievalSpan, GenerationSpan, TraceSession


def test_retrieval_span_score_length_mismatch():
    with pytest.raises(ValueError):
        RetrievalSpan(
            query="test",
            chunks=["a", "b"],
            scores=[0.9],  # length mismatch
            k_requested=2,
            k_returned=2,
        )


def test_generation_span_token_estimation():
    span = GenerationSpan(
        prompt="a" * 400,
        response="b" * 100,
        model="llama3.2",
    )
    # estimated from char count: 400//4=100, 100//4=25
    assert span.prompt_tokens == 100
    assert span.response_tokens == 25


def test_trace_session_has_retrieval():
    session = TraceSession(query="test")
    assert not session.has_retrieval

    session.add_retrieval_span(
        RetrievalSpan(
            query="test",
            chunks=["chunk"],
            scores=[0.8],
            k_requested=1,
            k_returned=1,
        )
    )
    assert session.has_retrieval
