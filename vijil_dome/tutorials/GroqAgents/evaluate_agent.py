import os
from dotenv import load_dotenv
import argparse
from news_agent import MyNewsBot, MyProtectedNewsBot
from vijil import Vijil
from vijil.local_agents.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
)

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate Groq Sample agent")
    parser.add_argument(
        "--protected",
        type=bool,
        required=False,
        default=False,
        help="Whether to evaluate the standard agent, or the one with Dome. Set to true to use dome.",
    )
    args = parser.parse_args()

    if args.protected:
        agent_name = "Protected News Agent"
    else:
        agent_name = "News Agent"

    # The expected output signature of this function depends on what your Agent needs
    def input_adapter(request: ChatCompletionRequest):
        # Extract whatever data you need from the request
        # Here we just take the last message content as the prompt
        return request.messages[-1]["content"]

    # The expected input signature of this function depends on your agent's output
    def output_adapter(agent_output: str) -> ChatCompletionResponse:
        # First create a message object
        # You can populate tool call and retrieval context if needed
        agent_response_message = ChatMessage(
            role="assistant",
            content=agent_output,
            tool_calls=None,
            retrieval_context=None,
        )

        # next create a choice object to support multiple completions if needed
        choice = ChatCompletionChoice(
            index=0, message=agent_response_message, finish_reason="stop"
        )

        # Finally, return the response
        return ChatCompletionResponse(
            model=agent_name,
            choices=[choice],
            usage=None,  # You can track usage as well
        )

    if args.protected:
        newsbot = MyProtectedNewsBot(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )

    else:
        newsbot = MyNewsBot(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )

    vijil = Vijil(
        api_key=os.getenv("VIJIL_API_KEY"),
    )

    local_agent = vijil.local_agents.create(
        agent_function=newsbot.answer_query,
        input_adapter=input_adapter,
        output_adapter=output_adapter,
    )

    vijil.local_agents.evaluate(
        agent_name=agent_name,  # This is the name of your agent to use in the evaluation
        evaluation_name=f"{agent_name} Security Evaluation",  # The name of your evaluation
        agent=local_agent,  # The LocalAgentExecutor you created earlier
        harnesses=["security"],  # The harnesses you wish to run
        rate_limit=50,  # maximum number of requests in the interval to send to your agent
        rate_limit_interval=1,  # The size of the interval for the rate_limit (in minutes)
    )
