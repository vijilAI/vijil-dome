import os
import asyncio
from dotenv import load_dotenv
from vijil_dome import Dome
from vijil_dome.integrations.mcp.wrapper import DomedMCPServer

if __name__ == "__main__":
    load_dotenv()

    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    config = {
        "mcpServers": {
            "search": {  # Remote search server
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
                "transport": "http",
            },
            "calculator": {  # Locally installed tool via library
                "command": "python",
                "args": ["-m", "mcp_server_calculator"],
                "transport": "stdio",
            },
            "encoding": {  # Locally running tool with structured outputs
                "command": "python",
                "args": ["-m", "dummy_jokes_tool"],
                "transport": "stdio",
            },
        }
    }

    dome = Dome("dome_config.toml")
    domed_server = DomedMCPServer(config, dome)
    asyncio.run(domed_server.initialize())
    domed_server.run(transport="http", host="0.0.0.0", port=8080)
