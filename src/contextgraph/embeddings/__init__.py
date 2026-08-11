"""Chunking and embedding.

``OpenAIEmbedder`` is safe to import without the openai package installed — the
SDK is imported inside the class, on first use, so the base install stays free
of provider dependencies and a host that brings its own ``Embedder`` never pays
for one it does not call.
"""

from __future__ import annotations

from .chunker import Chunk, chunk_text
from .openai import OpenAIEmbedder

__all__ = ["Chunk", "OpenAIEmbedder", "chunk_text"]
