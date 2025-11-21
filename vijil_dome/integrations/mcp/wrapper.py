from typing import Optional, Dict, Literal, Any
from mcp.types import TextContent
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from fastmcp.server.proxy import FastMCPProxy
from fastmcp.server.middleware import Middleware, MiddlewareContext
from vijil_dome import Dome

Transport = Literal["stdio", "http", "sse", "streamable-http"]

DEFAULT_TOOL_CALL_BLOCKED_MESSAGE = (
    "Sorry, your tool call request was flagged by Vijil Dome and cannot be processed."
)
DEFAULT_TOOL_RESPONSE_BLOCKED_MESSAGE = "Sorry, the response from the tool call was flagged by Vijil Dome and cannot be provided."


def transform_mcp_schema(schema: dict) -> dict:
    """
    Transform an MCP tool schema so that:
    - All properties become optional (i.e. removed from 'required')
    - A new boolean field 'blocked_by_guardrails' is added at root
    """

    def make_optional(node: dict):
        """Recursively remove 'required' from all object schemas."""
        if not isinstance(node, dict):
            return node

        # If this schema declares properties, remove 'required'
        if node.get("type") == "object" and "properties" in node:
            node.pop("required", None)

            for key, value in node["properties"].items():
                node["properties"][key] = make_optional(value)

        # Handle arrays with items
        if node.get("type") == "array" and "items" in node:
            node["items"] = make_optional(node["items"])

        # Support oneOf / anyOf / allOf
        for group in ("oneOf", "anyOf", "allOf"):
            if group in node:
                node[group] = [make_optional(x) for x in node[group]]

        return node

    # Deep copy to avoid mutating original
    new_schema = make_optional(dict(schema))

    # Inject guardrail field at the top level
    props = new_schema.setdefault("properties", {})
    props["blocked_by_guardrails"] = {"type": "boolean"}

    # Ensure root is object schema
    if new_schema.get("type") != "object":
        new_schema["type"] = "object"

    return new_schema


class DomedMCPServer:
    def __init__(
        self,
        mcp_server_config: Dict,
        dome: Dome,
        server_name: Optional[str] = None,
        tool_call_input_block_message: Optional[str] = None,
        tool_call_output_block_message: Optional[str] = None,
    ):
        self.dome = dome
        self.mcp_server_config = mcp_server_config
        self.tool_call_input_block_message = (
            tool_call_input_block_message or DEFAULT_TOOL_CALL_BLOCKED_MESSAGE
        )
        self.tool_call_output_block_message = (
            tool_call_output_block_message or DEFAULT_TOOL_RESPONSE_BLOCKED_MESSAGE
        )
        self.server_name = server_name or "Domed MCP Server"
        self.server = None  # type: Optional[FastMCPProxy]

    async def initialize(self):
        self.server = await self._create_guardrailed_server()

    async def _create_guardrailed_server(self) -> FastMCPProxy:
        dome_instance = self.dome
        tool_call_block_message = self.tool_call_input_block_message
        tool_response_block_message = self.tool_call_output_block_message

        class InputGuardrailMiddleware(Middleware):
            async def on_call_tool(self, context: MiddlewareContext, call_next):
                original_input = str(context.message)
                input_scan = await dome_instance.async_guard_input(original_input)
                if input_scan.flagged:
                    return ToolResult(
                        content=[
                            TextContent(type="text", text=tool_call_block_message)
                        ],
                    )
                    # raise ToolError(tool_call_block_message)
                result = await call_next(context)
                return result

        class OutputGuardrailMiddleware(Middleware):
            async def on_call_tool(self, context: MiddlewareContext, call_next):
                result = await call_next(context)
                output_scan = await dome_instance.async_guard_output(
                    str(result.structured_content)
                )
                if output_scan.flagged:
                    return ToolResult(
                        content=[
                            TextContent(type="text", text=tool_response_block_message)
                        ],
                    )
                    # raise ToolError(tool_response_block_message)
                return result

        proxy = FastMCP.as_proxy(self.mcp_server_config, name=self.server_name)

        # Disable forced output schemas to allow guardrail middleware to modify outputs freely
        # If we don't do this, guardrailed outputs that contain just the block message will fail schema validation
        # TBD: How can we keep this for better performance and retain compatibility with local stdio mcp servers?

        tools = await proxy.get_tools()
        for tool_name in tools:
            local_tool = tools[tool_name].copy()
            local_tool.output_schema = None
            proxy.add_tool(local_tool)

        proxy.add_middleware(InputGuardrailMiddleware())
        proxy.add_middleware(OutputGuardrailMiddleware())
        return proxy

    def run(
        self,
        transport: Transport | None = None,
        show_banner: bool = False,
        **transport_kwargs: Any,
    ):
        # Run the underlying fastMCPProxy server. Args are the same as FastMCPProxy.run()
        if self.server is None:
            raise RuntimeError(
                "Server not initialized. Call 'initialize' before running the server."
            )
        self.server.run(
            transport=transport, show_banner=show_banner, **transport_kwargs
        )
