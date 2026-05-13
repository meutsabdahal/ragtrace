from ragtrace.renderers.html import render_html
from ragtrace.session import GenerationSpan, RetrievalSpan, TraceSession


def test_render_html_escapes_dynamic_content():
    session = TraceSession(query="What happens with <script>alert(1)</script>?")
    session.add_retrieval_span(
        RetrievalSpan(
            query="What happens with <script>alert(1)</script>?",
            chunks=["chunk <img src=x onerror=alert(1)>"],
            scores=[0.9],
            k_requested=1,
            k_returned=1,
        )
    )
    session.add_generation_span(
        GenerationSpan(
            prompt="prompt <b>unsafe</b>",
            response="response <b>unsafe</b>",
            model="test-model",
        )
    )

    html = render_html(session)

    assert "What happens with &lt;script&gt;alert(1)&lt;/script&gt;?" in html
    assert "\\u003cimg src=x onerror=alert(1)>" in html
    assert "\\u003cb>unsafe\\u003c/b>" in html
