
from agents import Agent
from pydantic import BaseModel


class AgentAction(BaseModel):
    action: str
    amount: float
    reasoning: str


def create_agent(prompt: str, tools: list, model: str) -> Agent:
    return Agent(
        name="Trader",
        instructions=prompt,
        tools=tools,
        output_type=AgentAction,
        model=model
    )
