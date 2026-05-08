from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import uuid


@dataclass
class RetrievalSpan:
    query: str
    chunks: list[str]
    scores: list[float]
    k_requested: int
    k_returned: int
    latency_ms: float = 0.0
    diagnosis: list[str] = field(default_factory=list)

    def __post_init__(self):
        # normalise scores — some vector stores return distances
        # (lower = better), others return similarities (higher = better).
        # ragtrace assumes higher = better. Users are responsible for
        # converting distances to similarities before calling log_retrieval.
        if len(self.scores) != len(self.chunks):
            raise ValueError(
                f"scores length ({len(self.scores)}) must match "
                f"chunks length ({len(self.chunks)})"
            )


@dataclass
class GenerationSpan:
    prompt: str
    response: str
    model: str
    prompt_tokens: int = 0
    response_tokens: int = 0
    latency_ms: float = 0.0
    diagnosis: list[str] = field(default_factory=list)

    def __post_init__(self):
        # estimate token counts if not provided
        # rough approximation: 1 token ≈ 4 chars
        if self.prompt_tokens == 0:
            self.prompt_tokens = len(self.prompt) // 4
        if self.response_tokens == 0:
            self.response_tokens = len(self.response) // 4


@dataclass
class TraceSession:
    query: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    retrieval_spans: list[RetrievalSpan] = field(default_factory=list)
    generation_spans: list[GenerationSpan] = field(default_factory=list)
    total_latency_ms: float = 0.0
    _start_time: float = field(default_factory=time.perf_counter, repr=False)

    def add_retrieval_span(self, span: RetrievalSpan) -> None:
        self.retrieval_spans.append(span)

    def add_generation_span(self, span: GenerationSpan) -> None:
        self.generation_spans.append(span)

    def finalise(self) -> None:
        self.total_latency_ms = (time.perf_counter() - self._start_time) * 1000

    @property
    def has_retrieval(self) -> bool:
        return len(self.retrieval_spans) > 0

    @property
    def has_generation(self) -> bool:
        return len(self.generation_spans) > 0
