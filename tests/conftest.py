"""Test doubles and fixtures.

The fakes are deliberate: a deterministic embedder makes similarity a thing the
test *controls* rather than observes, which is the only way to assert on
threshold behaviour without paying a provider and accepting flakiness.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any

import pytest

DIM = 1536


class FakeEmbedder:
    """Deterministic embedder with controllable similarity.

    Identical text yields identical vectors (cosine 1.0). Text sharing a
    ``~``-prefixed group token yields a vector close to that group's centroid,
    so a test can place two inputs at a chosen similarity band.
    """

    dimensions = DIM
    model = "fake-embedding"

    def __init__(self) -> None:
        self.tokens_used = 0
        self.calls = 0

    @staticmethod
    def _unit(seed: str) -> list[float]:
        h = hashlib.sha256(seed.encode()).digest()
        raw = [(h[i % len(h)] - 128) / 128.0 for i in range(DIM)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def _vector(self, text: str) -> list[float]:
        # "~group~ rest of text" -> mostly the group vector, slightly perturbed.
        if text.startswith("~") and "~" in text[1:]:
            group, _, rest = text[1:].partition("~")
            base = self._unit(group)
            noise = self._unit(rest)
            mixed = [0.97 * b + 0.03 * n for b, n in zip(base, noise, strict=True)]
            norm = math.sqrt(sum(x * x for x in mixed)) or 1.0
            return [x / norm for x in mixed]
        return self._unit(text)

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        self.tokens_used += max(1, len(text) // 4)
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.tokens_used += sum(max(1, len(t) // 4) for t in texts)
        return [self._vector(t) for t in texts]


class FakeLLM:
    """Returns queued responses. Records every prompt it was given."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.prompts: list[dict[str, Any]] = []
        self.default_verdict = "distinct"

    async def complete(
        self, *, system: list[str], user: str, schema: type,
        max_tokens: int = 4096, temperature: float = 0.0,
    ) -> tuple[Any | None, dict[str, Any] | None]:
        self.prompts.append({"system": system, "user": user, "schema": schema})
        usage = {"input_tokens": 100, "output_tokens": 20, "model": "fake-llm"}
        if self.responses:
            return self.responses.pop(0), usage
        # Default: an adjudication asked for with nothing queued answers
        # "distinct", matching the library's own safe default.
        if schema.__name__ == "RelationshipVerdict":
            return schema(relationship=self.default_verdict, reason="fake"), usage
        return schema(nodes=[], edges=[]), usage


class RecordingMeter:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def record(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def meter() -> RecordingMeter:
    return RecordingMeter()


def _database_url() -> str | None:
    url = os.environ.get("CONTEXTGRAPH_TEST_DATABASE_URL") or os.environ.get(
        "DATABASE_URL"
    )
    if not url:
        return None
    return url.replace("postgresql://", "postgresql+asyncpg://", 1)


@pytest.fixture
async def session_factory():
    """Session factory against a live Postgres, or skip.

    Integration tests are skipped rather than mocked: the whole point of them
    is the SQL, and mocked SQL tests assert that strings equal strings.
    """
    url = _database_url()
    if not url:
        pytest.skip("Set DATABASE_URL to run integration tests")

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
async def clean_graph(session_factory):
    """A unique graph id, torn down afterwards."""
    import uuid

    from sqlalchemy import text

    graph_id = f"test-{uuid.uuid4().hex[:12]}"
    yield graph_id
    async with session_factory() as s:
        for table in (
            "cg_edges", "cg_nodes", "cg_segments",
            "cg_changesets", "cg_sources", "cg_runs",
        ):
            await s.execute(
                text(f"DELETE FROM {table} WHERE graph_id = :g"), {"g": graph_id}
            )
        await s.commit()
