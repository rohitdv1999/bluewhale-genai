"""MCP server: exposes the job tools to any MCP client (desktop assistants, IDE agents, ...).

Only READ tools are exposed. Anything that writes or sends goes through the
HTTP API's agent runs, where the human-approval gate lives.

    bluewhale-mcp            # stdio transport
"""

from __future__ import annotations

import json
from typing import Any

from .agents.job_match import build_job_tools
from .agents.tools import Permission
from .container import build_services, load_samples


def build_server() -> Any:
    from mcp.server.fastmcp import FastMCP  # optional: pip install .[mcp]

    services = build_services()
    if services.settings.is_local:
        load_samples(services.rag)
    registry = build_job_tools(services.rag)
    server = FastMCP("bluewhale")

    for name in registry.names():
        tool = registry.get(name)
        if tool is None or tool.permission != Permission.READ:
            continue

        def handler(arguments: dict[str, Any], _name: str = name) -> str:
            result = registry.run(_name, arguments)
            return json.dumps(result.output if result.ok else {"error": result.error}, default=str)

        server.add_tool(handler, name=tool.name, description=tool.description)
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
