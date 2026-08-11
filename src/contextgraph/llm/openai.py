"""GPT, via langchain-openai.

Identical contract to ``AnthropicLLM``. It exists so that ``StructuredLLM``
is demonstrably a seam and not a euphemism for one vendor: if the only
implementation of an interface is the one it was designed around, the
interface has not been tested.

Module name shadowing: this is ``contextgraph.llm.openai``, and
``langchain_openai`` imports the top-level ``openai`` SDK. Under absolute
imports (Python 3) that resolves to site-packages, not to this file.
"""

from __future__ import annotations

from typing import Any, Literal

DEFAULT_MODEL = "gpt-5"

StructuredMethod = Literal["function_calling", "json_schema", "json_mode"]


class OpenAILLM:
    """Structured completion against OpenAI chat models.

    ``api_key`` exists so a host can pass a per-tenant key (BYOK); omit it to
    fall back to the environment.

    ``method`` defaults to ``function_calling`` because it is the only mode
    every chat model in the fleet supports. ``json_schema`` is strictly
    enforced server-side and is the better choice where the deployed model
    supports it — it is a constructor argument rather than a hardcoded value
    precisely because that support is a property of the caller's model, not of
    this package.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        method: StructuredMethod = "function_calling",
        timeout: float | None = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._method: StructuredMethod = method
        self._timeout = timeout
        self._max_retries = max_retries
        # Same reasoning as the Anthropic adapter: each ChatOpenAI builds its
        # own HTTP client, and only these two settings vary per call.
        self._clients: dict[tuple[int, float], Any] = {}

    def _client(self, max_tokens: int, temperature: float) -> Any:
        key = (max_tokens, temperature)
        cached = self._clients.get(key)
        if cached is not None:
            return cached

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "OpenAILLM requires langchain-openai. "
                'Install it with: pip install "contextgraph[openai]"'
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self.model,
            # Aliased to max_completion_tokens by langchain-openai; the older
            # max_tokens field is rejected by the reasoning models.
            "max_tokens": max_tokens,
            # No model-aware filtering needed here, unlike the Anthropic
            # adapter: langchain-openai drops temperature itself for the models
            # that reject a non-default value (gpt-5 non-chat) and pins it to 1
            # for the o1 family.
            "temperature": temperature,
            "timeout": self._timeout,
            "max_retries": self._max_retries,
        }
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key

        client = ChatOpenAI(**kwargs)
        self._clients[key] = client
        return client

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
                "OpenAILLM requires langchain-core. "
                'Install it with: pip install "contextgraph[openai]"'
            ) from exc

        # One SystemMessage per block, in order — OpenAI takes several
        # system-role messages directly, no merging involved. Order still
        # matters for the same reason it does on Anthropic: OpenAI's automatic
        # caching is a prefix match, so the stable prefix goes first and
        # per-call content last.
        messages: list[Any] = [
            SystemMessage(content=block) for block in system if block and block.strip()
        ]
        messages.append(HumanMessage(content=user))

        client = self._client(max_tokens, temperature)

        # include_raw=True for the same reason as the Anthropic adapter: the
        # plain form returns the parsed model alone and discards the AIMessage
        # carrying usage metadata, which makes the call impossible to meter.
        # Shape: {"raw": AIMessage, "parsed": ... | None, "parsing_error": ... | None}.
        structured = client.with_structured_output(
            schema, include_raw=True, method=self._method
        )

        # Transport and auth failures propagate; only parse failures are
        # absorbed into a None result.
        result = await structured.ainvoke(messages)

        raw = result.get("raw") if isinstance(result, dict) else None
        usage = self._usage(raw)

        if not isinstance(result, dict):  # pragma: no cover - contract violation
            return None, usage
        if result.get("parsing_error") is not None:
            return None, usage
        return result.get("parsed"), usage

    def _usage(self, raw: Any) -> dict[str, Any] | None:
        if raw is None:
            return None
        metadata = getattr(raw, "usage_metadata", None)
        if not metadata:
            return None

        response_metadata = getattr(raw, "response_metadata", None) or {}
        # The served model, not the requested alias — "gpt-5" resolves to a
        # dated snapshot, and metering should record what actually ran.
        model = (
            response_metadata.get("model_name")
            or response_metadata.get("model")
            or self.model
        )
        # input_tokens here is the total prompt, cached tokens included. Hosts
        # that price cache hits separately can read
        # usage_metadata["input_token_details"]["cache_read"].
        return {
            "input_tokens": int(metadata.get("input_tokens") or 0),
            "output_tokens": int(metadata.get("output_tokens") or 0),
            "model": model,
        }


__all__ = ["DEFAULT_MODEL", "OpenAILLM", "StructuredMethod"]
