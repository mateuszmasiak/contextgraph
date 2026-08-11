"""OpenAI reference implementation of the ``Embedder`` protocol.

The whole of this module exists to satisfy one requirement from
``protocols.Embedder``: on failure, raise. Everything else — batching,
normalisation, token accounting — is ordinary plumbing.
"""

from __future__ import annotations

import math
import os
from typing import Any

from ..errors import EmbeddingUnavailableError, SchemaMismatchError

# text-embedding-3-* accept a `dimensions` parameter and return a truncated,
# no-longer-unit-length vector; ada-002 and most third-party OpenAI-compatible
# endpoints reject the parameter outright, so it is sent only when supported.
_SUPPORTS_DIMENSIONS_PREFIX = "text-embedding-3"


class OpenAIEmbedder:
    """Embeds via OpenAI. Never returns a vector it did not receive.

    Two error types, and the difference is the caller's next move:

    - ``EmbeddingUnavailableError`` — the provider did not give us usable
      vectors. Recoverable; re-run later.
    - ``SchemaMismatchError`` — the provider gave us vectors of a width the
      database cannot store. Fatal; retrying reproduces it exactly. Raised
      rather than folded into the first type because "try again in five
      minutes" is precisely the wrong response to a migration problem.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
        timeout: float = 30.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client
        self._tokens_used = 0
        # None until the provider has told us the true width. Checked once, on
        # the first response, so a misconfiguration fails on the first batch
        # instead of on the first INSERT.
        self._verified_dimensions: int | None = None

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if client is None and not self._api_key:
            # Fails at construction rather than at the first embed call, so a
            # deployment missing its key breaks at startup and not halfway
            # through structuring a source.
            raise EmbeddingUnavailableError(
                "No OpenAI API key: pass api_key= or set OPENAI_API_KEY. "
                "Refusing to construct an embedder that cannot embed."
            )

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                # The provider rejects empty input, and there is no honest
                # substitute: embedding a placeholder would store a vector that
                # describes text the source does not contain. chunk_text()
                # already drops empty chunks, so this is a caller bug.
                raise ValueError(
                    f"texts[{i}] is empty or whitespace-only; nothing to embed"
                )

        client = self._get_client()
        vectors: list[list[float]] = []

        # Sequential by design. Concurrent batches would finish out of order and
        # complicate partial-failure handling for a gain that provider rate
        # limits mostly take back.
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            vectors.extend(await self._embed_one_batch(client, batch))

        if len(vectors) != len(texts):
            raise EmbeddingUnavailableError(
                f"Embedder produced {len(vectors)} vectors for {len(texts)} texts"
            )
        return vectors

    async def _embed_one_batch(
        self, client: Any, batch: list[str]
    ) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": batch}
        if self.model.startswith(_SUPPORTS_DIMENSIONS_PREFIX):
            kwargs["dimensions"] = self.dimensions

        try:
            response = await client.embeddings.create(**kwargs)
        except Exception as exc:
            # Every provider failure lands here: auth, rate limit past retries,
            # timeout, connection reset, 5xx. All of them raise. Returning a
            # zero or random vector instead would be undetectable downstream — a
            # fabricated vector is indistinguishable from a real one once
            # stored, and it silently corrupts every similarity decision made
            # against it thereafter, including merges, which are irreversible.
            # Failing is recoverable: the raw source is retained and structuring
            # is re-runnable.
            raise EmbeddingUnavailableError(
                f"OpenAI embeddings failed for {len(batch)} texts "
                f"(model={self.model}): {type(exc).__name__}: {exc}"
            ) from exc

        data = getattr(response, "data", None)
        if not data:
            raise EmbeddingUnavailableError(
                f"OpenAI returned no embedding data for {len(batch)} texts"
            )

        if len(data) != len(batch):
            # Never zip a short response: it pairs every vector after the gap
            # with the wrong text, and the mispairing is invisible thereafter.
            raise EmbeddingUnavailableError(
                f"OpenAI returned {len(data)} embeddings for {len(batch)} texts"
            )

        # Order by the provider's own index rather than by position in the
        # response array — the API documents the index field precisely because
        # response order is not part of the contract.
        try:
            ordered = sorted(data, key=lambda item: item.index)
        except (AttributeError, TypeError) as exc:
            raise EmbeddingUnavailableError(
                f"OpenAI embedding items lack a usable index field: {exc}"
            ) from exc

        if [item.index for item in ordered] != list(range(len(batch))):
            raise EmbeddingUnavailableError(
                "OpenAI embedding indices are not a complete 0..n-1 range; "
                "cannot pair vectors with their texts"
            )

        usage = getattr(response, "usage", None)
        if usage is not None:
            self._tokens_used += int(getattr(usage, "total_tokens", 0) or 0)

        return [self._finalise(item.embedding, batch[item.index]) for item in ordered]

    def _finalise(self, raw: Any, text: str) -> list[float]:
        try:
            vector = [float(v) for v in raw]
        except (TypeError, ValueError) as exc:
            raise EmbeddingUnavailableError(
                f"OpenAI returned a non-numeric embedding: {exc}"
            ) from exc

        if self._verified_dimensions is None:
            if len(vector) != self.dimensions:
                raise SchemaMismatchError(
                    f"Model {self.model} returned {len(vector)}-dimensional "
                    f"vectors but this embedder is configured for "
                    f"{self.dimensions}, which must match the vector column the "
                    f"migration created. Fix the configuration, or migrate the "
                    f"column and re-embed every segment."
                )
            self._verified_dimensions = len(vector)
        elif len(vector) != self._verified_dimensions:
            raise EmbeddingUnavailableError(
                f"OpenAI returned a {len(vector)}-dimensional vector after "
                f"{self._verified_dimensions}; the batch is not internally "
                f"consistent"
            )

        norm = math.sqrt(math.fsum(v * v for v in vector))
        if norm == 0.0 or not math.isfinite(norm):
            # A zero or non-finite vector has no direction, so cosine distance
            # against it is undefined and pgvector will happily return it as a
            # neighbour of everything. Refuse rather than store it.
            raise EmbeddingUnavailableError(
                f"OpenAI returned a degenerate vector (norm={norm}) for a "
                f"{len(text)}-character text"
            )

        # L2-normalise so cosine and inner product agree and pgvector's <=>
        # behaves consistently. Required, not cosmetic: text-embedding-3-*
        # returns unit vectors only at native width, and a `dimensions`-reduced
        # response is truncated, hence no longer unit length.
        return [v / norm for v in vector]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            # Absolute import: inside contextgraph.embeddings.openai this
            # resolves to the installed openai package, not to this module.
            # Imported here rather than at module scope so the base package
            # installs without the provider SDK.
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "The openai package is not installed. "
                "Install it with: pip install 'contextgraph[openai]'"
            ) from exc

        # AsyncOpenAI, never OpenAI. The sync client blocks the caller's event
        # loop for the full HTTP round trip, stalling every other concurrent
        # request in the process — a 400ms embedding call becomes 400ms of
        # dead air for the whole server.
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            timeout=self._timeout,
            # Small on purpose. The SDK retries 429s and 5xx with backoff; a
            # large budget turns a provider outage into a run that holds its
            # database transaction open for minutes. The pipeline is
            # re-runnable, so failing early costs less than stalling.
            max_retries=self._max_retries,
        )
        return self._client

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        client = self._client
        if client is None:
            return
        closer = getattr(client, "close", None)
        if closer is not None:
            await closer()


__all__ = ["OpenAIEmbedder"]
