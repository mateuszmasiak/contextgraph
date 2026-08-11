"""Claude, via langchain-anthropic.

The reference ``StructuredLLM``. Two things in here are not obvious and both
were arrived at by reading provider behaviour rather than documentation:
``include_raw`` is mandatory if you ever want to bill for a call, and the
*shape* of the system prompt decides whether provider prompt caching works at
all.

Module name shadowing: this is ``contextgraph.llm.anthropic``, and
``langchain_anthropic`` imports the top-level ``anthropic`` SDK. Under
absolute imports (Python 3) that resolves to site-packages, not to this file.
"""

from __future__ import annotations

from typing import Any

# Sampling parameters were REMOVED — not deprecated, not ignored — from these
# model families: sending ``temperature`` returns 400 invalid_request_error.
# langchain-anthropic passes ``temperature`` straight into the request payload
# and does no model-aware filtering (unlike langchain-openai, which drops it
# for gpt-5), so the check has to live here or the default extraction path
# fails on the default model.
_NO_SAMPLING_PARAMS: tuple[str, ...] = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

DEFAULT_MODEL = "claude-opus-5"


class AnthropicLLM:
    """Structured completion against Claude.

    ``api_key`` exists so a host can pass a per-tenant key. Metering and
    billing are the host's business (see the ``Meter`` protocol); a library
    that reads only a process-wide ``ANTHROPIC_API_KEY`` forces every tenant
    onto one bill and makes BYOK impossible. Omit it to fall back to the
    environment.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float | None = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        # Constructing a ChatAnthropic builds a new HTTP client, so instances
        # are keyed by the only per-call settings that change and reused. The
        # pipeline uses a handful of distinct (max_tokens, temperature) pairs,
        # so this stays small; it is a connection-reuse cache, not a memo.
        self._clients: dict[tuple[int, float | None], Any] = {}

    def _client(self, max_tokens: int, temperature: float | None) -> Any:
        key = (max_tokens, temperature)
        cached = self._clients.get(key)
        if cached is not None:
            return cached

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "AnthropicLLM requires langchain-anthropic. "
                'Install it with: pip install "contextgraph[anthropic]"'
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self._timeout,
            "max_retries": self._max_retries,
        }
        # Passing api_key=None would override the env-var validator with an
        # empty secret, which fails at request time with a confusing auth error
        # rather than at construction.
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key

        client = ChatAnthropic(**kwargs)
        self._clients[key] = client
        return client

    def _temperature_for_model(self, temperature: float) -> float | None:
        """None means "omit the parameter entirely"."""
        if self.model.startswith(_NO_SAMPLING_PARAMS):
            # Dropped rather than raised: callers pass a low temperature asking
            # for near-determinism, and these models are near-deterministic
            # without it. Failing the call would make the package's own default
            # model unusable with its own default extraction config.
            return None
        return temperature

    async def complete(
        self,
        *,
        system: list[str],
        user: str,
        schema: type,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[Any | None, dict[str, Any] | None]:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "AnthropicLLM requires langchain-core. "
                'Install it with: pip install "contextgraph[anthropic]"'
            ) from exc

        # One SystemMessage per block, in order. Verified in langchain-anthropic:
        # consecutive SystemMessages are merged by _merge_messages and rendered
        # as Anthropic's `system` list-of-text-blocks, preserving order. That
        # ordering is the whole point — a stable prefix (ontology, instructions)
        # stays first and per-call content (the roster, the source) goes last,
        # so provider prompt caching, which is a prefix match, is not
        # invalidated by content that changes every call.
        #
        # Empty blocks are dropped: the API rejects a zero-length text block.
        messages: list[Any] = [
            SystemMessage(content=block) for block in system if block and block.strip()
        ]
        messages.append(HumanMessage(content=user))

        client = self._client(max_tokens, self._temperature_for_model(temperature))

        # include_raw=True is not optional. The plain form returns only the
        # parsed model and DISCARDS the underlying AIMessage — with it the
        # token counts, which is to say every call becomes unmeterable and a
        # parse failure becomes indistinguishable from an empty result. The
        # include_raw shape is
        # {"raw": AIMessage, "parsed": ... | None, "parsing_error": ... | None}.
        structured = client.with_structured_output(schema, include_raw=True)

        # Transport and auth failures propagate: they are the caller's to
        # retry, and swallowing them would silently produce an empty graph.
        result = await structured.ainvoke(messages)

        raw = result.get("raw") if isinstance(result, dict) else None
        usage = self._usage(raw)

        if not isinstance(result, dict):  # pragma: no cover - contract violation
            return None, usage
        if result.get("parsing_error") is not None:
            # The tokens were spent, so usage is returned regardless. The caller
            # treats a None parse as an empty result, not as an error.
            return None, usage
        return result.get("parsed"), usage

    def _usage(self, raw: Any) -> dict[str, Any] | None:
        if raw is None:
            return None
        metadata = getattr(raw, "usage_metadata", None)
        if not metadata:
            return None

        response_metadata = getattr(raw, "response_metadata", None) or {}
        # The response names the model actually served, which is what should be
        # metered — an alias like "claude-opus-5" resolves to a dated snapshot.
        model = (
            response_metadata.get("model_name")
            or response_metadata.get("model")
            or self.model
        )
        # langchain-anthropic folds cache-read and cache-creation tokens into
        # input_tokens (Anthropic reports them separately, and its raw
        # input_tokens excludes them). Cached tokens bill at a different rate,
        # so a host that prices precisely should read
        # usage_metadata["input_token_details"] itself; the Meter protocol
        # takes a single input count, so the total is what is reported here.
        return {
            "input_tokens": int(metadata.get("input_tokens") or 0),
            "output_tokens": int(metadata.get("output_tokens") or 0),
            "model": model,
        }


__all__ = ["DEFAULT_MODEL", "AnthropicLLM"]
