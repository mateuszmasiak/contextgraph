"""Reference ``StructuredLLM`` implementations.

Importing this package imports no provider SDK. Each adapter defers
``langchain_anthropic`` / ``langchain_openai`` to the first call, so
``from contextgraph.llm import AnthropicLLM, OpenAILLM`` works with only one
extra installed — and a missing dependency surfaces as a named ImportError
naming the extra to install, rather than at package import time as a failure
that has nothing to do with what the caller is actually using.

Neither is privileged. ``StructuredLLM`` is satisfied structurally, so a host
with its own gateway, cache or key-rotation layer supplies its own object and
never imports either of these.
"""

from __future__ import annotations

from contextgraph.llm.anthropic import AnthropicLLM
from contextgraph.llm.openai import OpenAILLM

__all__ = ["AnthropicLLM", "OpenAILLM"]
