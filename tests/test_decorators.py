import pytest
from ragtrace import trace, log_retrieval, log_generation
from ragtrace.collector import get_collector


def setup_function():
    get_collector().clear()


def test_sync_decorator_returns_value():
    @trace
    def my_func(query: str) -> str:
        return f"answer to: {query}"

    result = my_func("test query")
    assert result == "answer to: test query"


def test_sync_decorator_preserves_name():
    @trace
    def my_pipeline(query: str) -> str:
        return query

    assert my_pipeline.__name__ == "my_pipeline"


@pytest.mark.asyncio
async def test_async_decorator_returns_value():
    @trace
    async def async_pipeline(query: str) -> str:
        return f"async answer: {query}"

    result = await async_pipeline("async query")
    assert result == "async answer: async query"


def test_nested_traces_restore_outer_session():
    observed: dict[str, str | None] = {}

    @trace
    def inner() -> str:
        session = get_collector().get_current_session()
        assert session is not None
        observed["inner_session_id"] = session.session_id
        return session.session_id

    @trace
    def outer() -> str:
        session = get_collector().get_current_session()
        assert session is not None
        observed["outer_session_id"] = session.session_id
        inner_session_id = inner()
        current_session = get_collector().get_current_session()
        observed["after_inner_session_id"] = (
            None if current_session is None else current_session.session_id
        )
        return inner_session_id

    outer()

    assert observed["inner_session_id"] is not None
    assert observed["outer_session_id"] is not None
    assert observed["inner_session_id"] != observed["outer_session_id"]
    assert observed["after_inner_session_id"] == observed["outer_session_id"]


def test_log_retrieval_outside_trace_is_noop():
    # calling log_retrieval without @trace should not raise
    log_retrieval(query="test", chunks=["chunk"], scores=[0.8])


def test_trace_with_output_arg():
    import tempfile, os

    @trace(output="/tmp/test_ragtrace_report.html")
    def pipeline(query: str) -> str:
        log_retrieval(query=query, chunks=["chunk one"], scores=[0.75])
        log_generation(prompt="prompt", response="response", model="test")
        return "done"

    pipeline("test query")
    assert os.path.exists("/tmp/test_ragtrace_report.html")
    os.remove("/tmp/test_ragtrace_report.html")


def test_span_latencies_are_recorded(monkeypatch):
    ticks = iter([1.0, 1.25, 3.0, 3.5])

    monkeypatch.setattr("ragtrace.session.time.perf_counter", lambda: next(ticks))

    @trace
    def pipeline() -> object:
        log_retrieval(query="test", chunks=["chunk"], scores=[0.8])
        log_generation(prompt="prompt", response="response", model="test")
        return get_collector().get_current_session()

    session = pipeline()

    assert session is not None
    assert session.retrieval_spans[0].latency_ms == pytest.approx(250.0)
    assert session.generation_spans[0].latency_ms == pytest.approx(1750.0)
