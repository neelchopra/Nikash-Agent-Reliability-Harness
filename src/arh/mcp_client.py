"""Thin wrapper over the MCP python-sdk stdio client.

Connection/transport failures raise InfraError so the runner can retry them;
tool-level errors (result.isError) are returned as (text, ok=False) because
the AGENT caused them and must see and recover from them.
"""

from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from arh.errors import InfraError


class MCPSession:
    def __init__(self, params: StdioServerParameters):
        self._params = params
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "MCPSession":
        try:
            read, write = await self._stack.enter_async_context(stdio_client(self._params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
        except Exception as e:
            await self._stack.aclose()
            raise InfraError(f"MCP session startup failed: {e}") from e
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def list_tools(self) -> list:
        try:
            result = await self._session.list_tools()
        except Exception as e:
            raise InfraError(f"MCP list_tools failed: {e}") from e
        return result.tools

    async def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as e:
            raise InfraError(f"MCP transport error calling {name}: {e}") from e
        text = "\n".join(
            block.text for block in result.content if getattr(block, "text", None) is not None
        )
        return text, not result.isError
