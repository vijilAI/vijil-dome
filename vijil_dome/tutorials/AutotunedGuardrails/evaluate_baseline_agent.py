import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from vijil import Vijil
from vijil.agents.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
)


# The expected output signature of this function depends on what your Agent needs
def evaluation_input_adapter(request: ChatCompletionRequest):
    # Extract the message data from the request
    message_content = ""
    for message in request.messages:
        message_content += message["content"] + "\n"
    return message_content


# The expected input signature of this function depends on your agent's output
def evaluation_output_adapter(agent_output: str) -> ChatCompletionResponse:
    # First create a message object
    # You can populate tool call and retrieval context if needed
    agent_response_message = ChatMessage(
        role="assistant", content=agent_output, tool_calls=None, retrieval_context=None
    )

    # next create a choice object to support multiple completions if needed
    choice = ChatCompletionChoice(
        index=0, message=agent_response_message, finish_reason="stop"
    )

    # Finally, return the response
    return ChatCompletionResponse(
        model="my-desktop-assistant",  # You can set this to your model name
        choices=[choice],
        usage=None,  # You can track usage as well
    )


# Make sure to set your OpenAI API key as an environment variable
class MyDesktopAssistant:
    def __init__(self, system_prompt_path: str, api_key: Optional[str]):
        # Initialize your agent here
        self.chat_model = ChatOpenAI(model="gpt-4.1", streaming=False, api_key=api_key)
        with open(system_prompt_path, "r") as f:
            self.system_prompt = f.read()

    async def invoke(self, prompt: str) -> str:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = await self.chat_model.ainvoke(messages)
        return response.content


if __name__ == "__main__":
    load_dotenv()

    vijil = Vijil(
        api_key=os.getenv("VIJIL_API_KEY"),
    )

    # You can use whatever system prompt you like here
    my_agent = MyDesktopAssistant(
        system_prompt_path="desktop_assistant_prompt.txt",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    local_agent = vijil.agents.create(
        agent_function=my_agent.invoke,
        input_adapter=evaluation_input_adapter,
        output_adapter=evaluation_output_adapter,
    )

    vijil.agents.evaluate(
        agent_name="My Desktop Assistant",  # The is the name of your agent
        evaluation_name="Desktop Assistant Evaluation",  # The name of your evaluation
        agent=local_agent,  # The LocalAgentExecutor you created earlier
        harnesses=["safety", "security"],  # harnesses you wish to run
        rate_limit=500,  # Maximum number of requests in the interval
        rate_limit_interval=1,  # The size of the interval for the rate_limit (in minutes)
    )
