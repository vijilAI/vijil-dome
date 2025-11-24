from dotenv import load_dotenv
from typing import Optional

# This example agent is built using langgraph, but domed MCP servers are compatible with any MCP client
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI


load_dotenv()

AGENT_GRAPH: Optional[StateGraph] = None

model = ChatOpenAI(
    model="gpt-4o-mini",
)

# Our domed MCP server config
conf = {
    "safetools": {
        "url": "http://localhost:8080/mcp",
        "transport": "streamable_http",
    },
}

client = MultiServerMCPClient(conf)


async def build_graph():
    # First, fetch the tools from the MCP client
    # Note that this is an async operation, so it needs to be done within an async function
    tools = await client.get_tools()

    # Next, define a function that calls the model with the tools
    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    # Finally, we build the graph. This is a simple two-node loop between the model and the tools.
    # This allows the agent to call tools as needed, including multiple tools in sequence.
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    return graph


async def get_agent_response(query: str):
    input_request = {"messages": query}

    # Instead of building the graph every time, we build it once and cache it.
    # This speeds up subsequent invocations.
    global AGENT_GRAPH
    if AGENT_GRAPH is None:
        AGENT_GRAPH = await build_graph()

    # Invoke the app
    result = await AGENT_GRAPH.ainvoke(input_request)
    final_response = result
    return final_response


if __name__ == "__main__":
    import asyncio

    prompts = [
        "What is 1234 multiplied by 5678?",
        "What is 2g1c? Search the internet for a latest answer",
        "Tell me a joke about bees",
    ]

    for prompt in prompts:
        response = asyncio.run(get_agent_response(prompt))
        print("Prompt:", prompt, "\nResponse:", response)
        print("---" * 10)
