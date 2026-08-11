"""MCP server exposing the graph as agent tools.

Requires the ``mcp`` extra: ``pip install contextgraph[mcp]``.
Run with ``contextgraph-mcp`` or ``python -m contextgraph.mcp.server``.
"""

from contextgraph.mcp.server import TOOLS, GraphMCPServer, main

__all__ = ["TOOLS", "GraphMCPServer", "main"]
