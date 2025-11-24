# Adding Guardrails to MCP servers

In addition to protecting agents, Dome can be used to add guardrails around MCP servers as well. Dome uses FastMCP under the hood to create a guardrailed proxy that can protect MCP servers from rogue agents, and agents from the outputs of MCP servers. 

Setting up a guardrailed MCP server is very straightforward, as shown in the snippet below

```
import asyncio

from vijil_dome import Dome
from vijil_dome.integrations.mcp.wrapper import DomedMCPServer

if __name__ == "__main__":
    config = {
        "mcpServers": {
            # Your MCP server config
        }
    }

    # Create a Dome instance
    dome = Dome()

    # Create an MCP server with the config + dome
    domed_server = DomedMCPServer(config, dome)

    # Initialize it 
    asyncio.run(domed_server.initialize())

    # Run the server! 
    domed_server.run(transport="http", 
                    host="0.0.0.0", 
                    port=8080)

    # or, just do domed_server.run() if you want to use stdio

```
**Note** : Input guardrails for MCP servers should protect the server from potentially risky LLM outputs, while output guardrails for MCP servers should protect downstream LLMs. As such, make sure you configure dome instances for MCP server protection appropriately, with things like prompt injection guards in the output guardrails as well. 

## Running this Example

### Overview
This example demonstrates how you can use Dome to protect your agents from MCP servers that run remotely and locally, either within your code base or as a separate module. 

We'll be using the Tavily MCP server as an example of a remote MCP server, a simple calculator MCP server, and a dummy malicious MCP server that we'll run locally.

### Setup

Check out the contents of this folder and set up a new virtual environment. 

```bash
python -m venv dome-tutorials
# Activate the virtual environment
# macOS/Linux:
source dome-tutorials/bin/activate
# Windows:
dome-tutorials\Scripts\activate

python -m pip install -r requirements.txt
```

Next, create a new .env file with the following environment variables. You can get
```
OPENAI_API_KEY = <Your OpenAI key>
TAVILY_API_KEY = <Your Tavily API key>
```

### Running the server

In one process, run 
```
python -m example_server
```

This will run the Domed MCP server on your localhost on port 8000.

To interact with this server, you can use 

```
python -m example_client_queries
```
To see examples of how to use the domed MCP server. The examples will print out the entire output message, which will include the outputs from the LLM as well as the intermediate tool calls that it performed. 

