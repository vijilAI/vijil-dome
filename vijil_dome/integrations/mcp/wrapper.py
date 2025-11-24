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


# In order to deal with tool calls that expect a structured output, there are two possible solutions
# 1. We modify the output schema to set all the fields as optional, and include guardrail info,
# 2. We create a minimal valid object that satisfies the required fields, and include guardrail info.
# While 1, is more ideal, it breaks JSON RPC communication with tools that run locally via stdio transport,
# Hence, we go for the second option.
# TODO: Revisit option 1, and see if we can make it work
def _make_minimal_object(schema: Optional[dict[str, Any]]):
    """
    Given an MCP-style JSON schema dict, return a minimally valid object
    that satisfies required fields. Optional fields are omitted.
    """

    if schema is None:
        return None

    schema_type = schema.get("type")

    # --- If oneOf / anyOf exists, use the first branch ---
    if "oneOf" in schema:
        return _make_minimal_object(schema["oneOf"][0])
    if "anyOf" in schema:
        return _make_minimal_object(schema["anyOf"][0])

    # --- Base types ---
    if schema_type == "string":
        return ""
    if schema_type == "number":
        return 0
    if schema_type == "integer":
        return 0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None

    # --- Enum ---
    if "enum" in schema:
        return schema["enum"][0]

    # --- Array ---
    if schema_type == "array":
        # minimal array is empty
        return []

    # --- Object ---
    if schema_type == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])

        output = {}
        for field in required:
            field_schema = props.get(field, {})
            output[field] = _make_minimal_object(field_schema)

        return output

    # Unknown fallback
    return None


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
                    # Dome is triggered. Create a minimal response object that satisfies output typecast
                    tool_in_ctx = await context.fastmcp_context.fastmcp.get_tool(
                        context.message.name
                    )
                    tool_structured_output = tool_in_ctx.output_schema
                    minimal_output = _make_minimal_object(tool_structured_output)
                    # Now add in the guardrail info
                    minimal_output["blocked_by_guardrails"] = True
                    minimal_output["guardrail_message"] = tool_call_block_message
                    return ToolResult(
                        content=[
                            TextContent(type="text", text=tool_call_block_message)
                        ],
                        structured_content=minimal_output,
                    )
                result = await call_next(context)
                return result

        class OutputGuardrailMiddleware(Middleware):
            async def on_call_tool(self, context: MiddlewareContext, call_next):
                result = await call_next(context)
                output_scan = await dome_instance.async_guard_output(
                    str(result.structured_content)
                )
                if output_scan.flagged:
                    # Dome is triggered. Create a minimal response object that satisfies output typecast
                    tool_in_ctx = await context.fastmcp_context.fastmcp.get_tool(
                        context.message.name
                    )
                    tool_structured_output = tool_in_ctx.output_schema
                    minimal_output = _make_minimal_object(tool_structured_output)
                    minimal_output["blocked_by_guardrails"] = True
                    minimal_output["guardrail_message"] = tool_response_block_message
                    return ToolResult(
                        content=[
                            TextContent(type="text", text=tool_response_block_message)
                        ],
                        structured_content=minimal_output,
                    )
                return result

        proxy = FastMCP.as_proxy(self.mcp_server_config, name=self.server_name)
        # Add guardrail middlewares
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
