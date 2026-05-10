from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from ragtrace.session import TraceSession, RetrievalSpan, GenerationSpan
from ragtrace.config import TracerConfig

console = Console()


def _score_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def _score_signal(score: float) -> str:
    if score >= 0.7:
        return "✓"
    if score >= 0.5:
        return "⚠"
    return "✗"


def render_retrieval(span: RetrievalSpan, config: TracerConfig) -> None:
    console.print(
        f" [bold]Retrieval[/bold]  "
        f"[dim]{span.latency_ms:.0f}ms  "
        f"k={span.k_returned}/{span.k_requested}[/dim]"
    )

    if config.show_chunks:
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        table.add_column("Chunk", style="dim", max_width=config.max_chunk_preview)
        table.add_column("Score", justify="right", width=6)
        table.add_column("", width=4)

        for chunk, score in zip(span.chunks, span.scores):
            preview = chunk.replace("\n", " ")[: config.max_chunk_preview]
            if len(chunk) > config.max_chunk_preview:
                preview += "..."
            color = _score_color(score)
            signal = _score_signal(score)
            table.add_row(
                preview,
                f"[{color}]{score:.2f}[/{color}]",
                f"[{color}]{signal}[/{color}]",
            )
        console.print(table)

    for diagnosis in span.diagnosis:
        icon = (
            "⚠"
            if any(
                w in diagnosis.lower()
                for w in ["low", "padding", "gap", "noise", "small"]
            )
            else "✓"
        )
        style = "yellow" if icon == "⚠" else "green"
        console.print(f" [{style}]{icon} {diagnosis}[/{style}]")
    console.print()


def render_generation(span: GenerationSpan, config: TracerConfig) -> None:
    console.print(
        f" [bold]Generation[/bold]  "
        f"[dim]{span.latency_ms:.0f}ms  "
        f"model={span.model}[/dim]"
    )
    console.print(
        f" [dim]Prompt tokens: {span.prompt_tokens}  "
        f"Response tokens: {span.response_tokens}[/dim]"
    )

    if config.show_prompt:
        preview = span.prompt[:200].replace("\n", " ")
        if len(span.prompt) > 200:
            preview += "..."
        console.print(f" [dim]Prompt: {preview}[/dim]")

    console.print()
    console.print(f" [bold]Response:[/bold]")
    for line in span.response.splitlines():
        console.print(f"  {line}")
    console.print()

    for diagnosis in span.diagnosis:
        icon = (
            "⚠"
            if any(
                w in diagnosis.lower()
                for w in ["hedging", "short", "low", "ignored", "failure"]
            )
            else "✓"
        )
        style = "yellow" if icon == "⚠" else "green"
        console.print(f" [{style}]{icon} {diagnosis}[/{style}]")
    console.print()


def render_session(session: TraceSession, config: TracerConfig) -> None:
    console.rule(
        Text(f" Query: {session.query[:80]} ", style="bold"),
        style="dim",
    )
    console.print()

    for span in session.retrieval_spans:
        render_retrieval(span, config=config)

    for span in session.generation_spans:
        render_generation(span, config=config)

    console.print(f" [dim]Total latency: {session.total_latency_ms:.0f}ms[/dim]")
    console.rule(style="dim")
    console.print()
