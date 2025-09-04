import os
import argparse
from vijil import Vijil
from vijil_dome import Dome
from dotenv import load_dotenv
from evaluate_baseline_agent import (
    MyDesktopAssistant,
    evaluation_input_adapter,
    evaluation_output_adapter,
)


# Create a protected agent class that invokes the original agent using Dome
class ProtectedAgent:
    def __init__(self, agent: MyDesktopAssistant, dome: Dome):
        self.agent = agent
        self.dome = dome

    async def guardrailed_invoke(self, prompt: str) -> str:
        # First, guard the input
        input_guard_result = await self.dome.async_guard_input(prompt)
        if input_guard_result.flagged:
            # This is our custom message when we trigger the input guardrails that protect the agent
            return "Input was flagged by Vijil Dome. This request violates the agent's usage policy"

        # If input is clean, invoke the agent
        agent_response = await self.agent.invoke(prompt)

        # Now, guard the output
        output_guard_result = await self.dome.async_guard_output(agent_response)
        if output_guard_result.flagged:
            # This is our custom message when we trigger the output guardrails that protect the agent
            return "This output was blocked by Vijil Dome."

        return agent_response


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run Vijil Dome Evaluation")
    parser.add_argument(
        "--evaluation-id",
        type=str,
        required=True,
        help="The evaluation ID for Dome.create_from_vijil_evaluation",
    )
    parser.add_argument(
        "--latency-threshold",
        type=float,
        required=False,
        help="The latency threshold (in seconds)",
        default=None,
    )
    args = parser.parse_args()

    if os.getenv("VIJIL_API_KEY") is None:
        raise ValueError("Vijil API key was not set. Please set it.")

    vijil = Vijil(
        api_key=os.getenv("VIJIL_API_KEY"),
    )

    dome = Dome.create_from_vijil_evaluation(
        evaluation_id=args.evaluation_id,
        api_key=os.getenv("VIJIL_API_KEY"),
        latency_threshold=args.latency_threshold,
    )

    my_baseline_agent = MyDesktopAssistant(
        system_prompt_path="desktop_assistant_prompt.txt",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    my_protected_agent = ProtectedAgent(my_baseline_agent, dome)

    local_agent = vijil.agents.create(
        agent_function=my_protected_agent.guardrailed_invoke,
        input_adapter=evaluation_input_adapter,
        output_adapter=evaluation_output_adapter,
    )

    vijil.agents.evaluate(
        agent_name="My Protected Desktop Assistant",  # The is the name of your agent
        evaluation_name="Protected Desktop Assistant Evaluation",  # The name of your evaluation
        agent=local_agent,  # The LocalAgentExecutor you created earlier
        harnesses=["safety", "security"],  # harnesses you wish to run
        rate_limit=50,  # Maximum number of requests in the interval
        rate_limit_interval=1,  # The size of the interval for the rate_limit (in minutes)
    )
