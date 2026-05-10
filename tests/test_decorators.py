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
