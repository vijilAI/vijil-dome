from openai import AsyncOpenAI
from vijil_dome import Dome


class MyNewsBot:
    def __init__(
        self,
        groq_api_key: str,
        tavily_api_key: str,
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.1,
        top_p: float = 0.4,
    ):
        self.client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
        )

        self.tools = [
            {
                "type": "mcp",
                "server_url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}",
                "server_label": "tavily",
                "require_approval": "never",
            }
        ]

        self.instructions = "You are a helpful AI assistant whose purpose is to provide bite sized digestible news about AI startups and developments. \
            Provide upto 5 stories regarding the topic asked by the user, and cover what the news item is, why its important, and what its implications are."
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    async def answer_query(self, query_string: str):
        try:
            response = await self.client.responses.create(
                model=self.model,
                input=query_string,
                tools=self.tools,
                stream=False,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            #  Get response content from responses API
            content = (
                response.output_text
                if hasattr(response, "output_text")
                else str(response)
            )

            return content
        except Exception:
            return "Sorry, an internal error occurred. Please try again."


class MyProtectedNewsBot:
    def __init__(
        self,
        groq_api_key: str,
        tavily_api_key: str,
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.1,
        top_p: float = 0.4,
    ):
        self.dome = Dome()
        self.news_bot = MyNewsBot(
            groq_api_key, tavily_api_key, model, temperature, top_p
        )

        self.input_blocked_message = "I'm sorry, but this request violates my usage guidelines. I cannot respond to this request."
        self.output_blocked_message = "I'm sorry, but this content is in violation of my usage guidelines. I cannot respond to this request."

    async def answer_query(self, query_string: str):
        input_scan = await self.dome.async_guard_input(query_string)
        if not input_scan.is_safe():
            return self.input_blocked_message
        agent_output = await self.news_bot.answer_query(query_string)
        output_scan = await self.dome.async_guard_output(agent_output)
        if not output_scan.is_safe():
            return self.output_blocked_message
        return output_scan.response_string
