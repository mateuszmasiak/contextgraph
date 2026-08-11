"""The seams. Everything host-specific enters through one of these.

contextgraph owns the graph — its schema, its resolution rules, its temporal
semantics, its write gate. It deliberately owns none of the following, because
every host application already has opinions about them:

- which embedding model you pay for
- which LLM you call and with whose key
- how (or whether) you bill for that
- what your canonical entities are

Each is a Protocol here, satisfied structurally — no base class to inherit, no
registry to configure. Reference implementations live in ``contextgraph.embeddings``
and ``contextgraph.llm``; swapping one is a constructor argument, not a fork.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Two hard requirements, both learned the hard way:

    1. ``dimensions`` must match the vector column the migration created. The
       schema pins it at construction time; a mismatch is a startup error, not
       a silent truncation.
    2. On failure this MUST raise. Returning a random or zero vector is the
       single most destructive thing an implementation can do here — a bad
       vector is indistinguishable from a good one once stored, and it
       silently corrupts every similarity decision made against it forever.
       Failing is recoverable; the raw source is retained and structuring is
       re-runnable.
    """

    dimensions: int

    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return exactly one vector per input, in order.

        Implementations must verify the count rather than zipping blindly: a
        short response otherwise pairs each vector with the wrong text.
        """
        ...

    @property
    def tokens_used(self) -> int:
        """Provider-reported tokens since construction, for metering."""
        ...


@runtime_checkable
class StructuredLLM(Protocol):
    """Calls a model and returns a validated object of the requested schema.

    ``complete`` must return ``(parsed, usage)``. ``usage`` carries provider
    token counts (``input_tokens`` / ``output_tokens`` / ``model``) or is None
    when unavailable — returning it is what makes metering possible at all.
    Many structured-output wrappers discard usage metadata by default; if
    yours does, request the raw response alongside the parsed value.

    ``parsed`` may be None when the model returned something unparseable. The
    caller treats that as an empty result, never as an error to swallow.
    """

    async def complete(
        self,
        *,
        system: list[str],
        user: str,
        schema: type,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[Any | None, dict[str, Any] | None]: ...


@runtime_checkable
class Meter(Protocol):
    """Reports what a run spent. Optional — the default is a no-op.

    Called once per run, after the work, with the accumulated spend. It must
    not raise: the tokens are already spent, so a metering failure must never
    roll back real work. It is also called for FAILED runs, because the
    provider was still paid for whatever was consumed before the failure.
    """

    async def record(
        self,
        *,
        tenant_id: str,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        embed_tokens: int,
        embed_model: str,
    ) -> None: ...


@runtime_checkable
class CanonicalResolver(Protocol):
    """Resolves an extracted entity against the host's own system of record.

    This is the seam that keeps contextgraph honest about ownership. If your
    application already has a ``screens`` table, that table — not the graph —
    is the authority on what screens exist. Implement this and an extracted
    "checkout page" resolves to your canonical "Checkout Screen" instead of
    becoming a second, competing record of the same thing.

    Return None when nothing matches (the common case). Otherwise return a
    dict with at least ``title``; include ``node_id`` if a graph node already
    represents this entity, and ``ref`` for any host identifier you want
    carried on the node as ``properties.source_entity``.

    IMPORTANT: return a human-readable ``title`` or None — never an opaque id
    or slug. Whatever you return becomes the node's title, and is then offered
    back to the extractor as canonical vocabulary, so an id returned here
    propagates permanently.
    """

    async def resolve(
        self, *, graph_id: str, node_type: str, title: str
    ) -> dict[str, Any] | None: ...


class NullMeter:
    """Default meter: records nothing."""

    async def record(self, **_: Any) -> None:  # noqa: D102
        return None


class NullCanonicalResolver:
    """Default resolver: the graph is the only system of record."""

    async def resolve(self, **_: Any) -> dict[str, Any] | None:  # noqa: D102
        return None
