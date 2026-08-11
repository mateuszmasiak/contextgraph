"""The failure modes, named.

A pipeline that turns text into stored beliefs has exactly three interesting
answers to "what now?": retry it, fix the configuration, or fix the code. The
types below exist so a caller can tell those apart without parsing message
strings. Anything raised deliberately by this library derives from
``ContextGraphError``; anything else escaping is a bug worth reporting.

The distinction that matters most is between *recoverable* and *fatal*.
Recoverable failures leave the raw source intact and the run marked failed, so
the work can simply be run again — that is the whole reason sources are
retained verbatim. Fatal failures will produce the identical outcome on every
retry, so retrying them only converts a clear error into a slow one.
"""

from __future__ import annotations


class ContextGraphError(Exception):
    """Base for every error this library raises on purpose.

    Catch this at the boundary of a background job to keep a single bad source
    from taking down a worker. Catching it around an individual operation to
    continue as if the operation succeeded is the one thing not to do: the
    subclasses below are raised precisely at the points where continuing would
    write something untrue into the graph.
    """


class EmbeddingUnavailableError(ContextGraphError):
    """Raised when a vector could not be obtained.

    Sources: no API key configured, the provider client library is not
    installed, the provider failed after its retries are exhausted, or the
    response was structurally unusable (wrong count, zero-magnitude vector).

    It is never raised *alongside* a substitute vector, and no caller should
    supply one on catching it. A fabricated vector is indistinguishable from a
    real one once stored and silently corrupts every similarity decision made
    against it thereafter — including merge decisions, which are irreversible.

    Recoverable. The raw source is retained and structuring is re-runnable, so
    the correct response is to mark the run failed and try again later. Meter
    the run anyway: the provider was still paid for whatever was consumed
    before the failure.
    """


class ExtractionError(ContextGraphError):
    """Raised when the extraction step could not complete.

    This means the model was not successfully consulted — transport failure,
    auth failure, exhausted retries, or input that cannot be sent at all. It
    does *not* mean the model found nothing: a well-formed response containing
    no nodes is an empty result, and an unparseable response is treated as an
    empty result too, per the ``StructuredLLM`` contract. Empty results are
    recorded, not raised.

    Recoverable. Set ``cg_sources.extraction_status = 'failed'``, keep the
    message in ``extraction_error``, and leave ``raw_content`` untouched so a
    later run can retry the same source.
    """


class SchemaMismatchError(ContextGraphError):
    """Raised when the running configuration disagrees with the migrated schema.

    The case that motivates it: an embedder whose ``dimensions`` differ from
    the width of the ``vector(N)`` column the migration created. Postgres would
    reject the insert anyway, but only after a paid embedding call and only for
    the unlucky row — so the check runs against the provider's first response
    and fails the whole run at once. Missing tables and a missing ``vector``
    extension raise it too.

    Fatal. Retrying re-runs the same mismatch. Either correct
    ``Config.embedding_dimensions`` and the embedder to agree, or migrate the
    column and re-embed every segment — a dimension change invalidates every
    stored vector, since distances between vectors of different models are
    meaningless even at equal width.
    """


class ScopeViolationError(ContextGraphError):
    """Raised when an operation would cross a tenant or graph boundary.

    Raised when a supplied node, edge or changeset id resolves to a row whose
    ``tenant_id``/``graph_id`` differ from the ones the caller is operating
    under — a changeset op naming a node outside its own graph, a retrieval
    filter that would leak rows across tenants.

    Fatal, and never to be worked around by widening the scope until the call
    passes. Treat it as a caller bug, or — if the ids came from an agent or an
    end user rather than from your own code — as an attempted access to another
    tenant's data, and log it accordingly.
    """


class ConcurrentModificationError(ContextGraphError):
    """Raised when applying a changeset against state that has since changed.

    Changesets are proposed against a read of the graph and applied later,
    possibly after human review. If a targeted node was merged, superseded or
    edited in between, applying the original ops would silently undo whoever
    got there first.

    Recoverable, but not by retrying the same payload — that reproduces the
    lost update. Re-read the current state, re-derive the operations against
    it, and re-submit. If the changeset was human-reviewed, it needs reviewing
    again: the thing that was approved is no longer the thing that would be
    applied.
    """


__all__ = [
    "ConcurrentModificationError",
    "ContextGraphError",
    "EmbeddingUnavailableError",
    "ExtractionError",
    "SchemaMismatchError",
    "ScopeViolationError",
]
