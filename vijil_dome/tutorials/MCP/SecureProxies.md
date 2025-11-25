# Creating and protecting secure MCP gateways

In addition to protecting your agents from MCP servers that may produce malicious content, you can set up secure authorization and MCP server monitoring via [Obot](https://obot.ai/). Dome makes it easy to create a secure MCP gateway and wrap guardrails around it. 

To create a secure proxy, [first install and set up Obot](https://docs.obot.ai/installation/overview). After that, you can generate a secure proxy directly within code

```python
from vijil_dome import Dome
from vijil_dome.integrations.mcp import DomedMCPServer
from vijil_dome.integrations.mcp.obot import ObotClient

config = {
    "mcpServers": {
        "tavily": {  # Example remote server
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
            "transport": "http",
        },
        "everything": { # Example Locally running sever
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-everything"]
        }
    }
}

obot_client = ObotClient(<Your Obot URL>, <Your Obot Token>)

# Create a safe config that uses Obot to manage the MCP gateway
safe_config = obot_client.create_secure_mcp_config(config)

# safe_config is a ready to use MCP server configuration
# {
#     'mcpServers': {
#        'tavily': {
#             'url': '<Your Obot URL>/mcp-connect/<id...>', 
#             'transport': 'streamable_http'
#         }, 
#         'everything': {
#             'url': '<Your Obot URL>/mcp-connect/<id...>', 
#             'transport': 'streamable_http'
#        }
#    }
#}

# You can now protect the MCP gateway with Dome

dome = Dome()
domed_server = DomedMCPServer(config, dome)
asyncio.run(domed_server.initialize())

domed_server.run()

```
