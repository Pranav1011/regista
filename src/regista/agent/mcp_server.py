"""Stdio MCP server exposing Regista's read-only match tools.

Run: ``regista mcp`` (or ``python -m regista.agent.mcp_server``). Stores are read
from ``data/store`` (override with REGISTA_DATA_DIR or --store-root).
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Callable
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
from mcp.types import ToolAnnotations

from regista import io
from regista.agent.tools import TOOL_NAMES, Toolbox, ToolError

INSTRUCTIONS = (
    "Regista answers tactical questions about football matches from precomputed tracking "
    "analytics. Every tool is read-only and returns evidence (match, period, frame range, "
    "match clock). Times are match clocks such as '62:00' or '45+2:00'. Matches are named "
    "'<source>/<match_id>', e.g. 'metrica/3'. Report numbers only as the tools return them, "
    "cite the evidence, and say when something is not available."
)


def _user_errors(fn: Callable) -> Callable:
    """Pass Regista's user-facing ToolError messages through to the MCP client."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError as e:
            raise MCPToolError(str(e)) from e

    return wrapper


def build_server(store_root: Path | None = None) -> MCPServer:
    toolbox = Toolbox(store_root or io.data_dir() / "store")
    server = MCPServer(name="regista", instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
    server.add_tool(
        toolbox.matches,
        name="list_matches",
        description="Matches with a store.",
        annotations=read_only,
    )
    for name in TOOL_NAMES:
        fn = getattr(toolbox, name)
        server.add_tool(_user_errors(fn), name=name, description=fn.__doc__, annotations=read_only)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Regista MCP server (stdio)")
    parser.add_argument("--store-root", type=Path, default=None)
    args = parser.parse_args()
    build_server(args.store_root).run("stdio")


if __name__ == "__main__":
    main()
