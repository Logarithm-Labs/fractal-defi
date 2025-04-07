from pydantic import BaseModel
from typing import List
from agents import Agent


class AgentAction(BaseModel):
    action: str
    amount: float
    reasoning: str


def create_agent(prompt: str, tools: List, model: str) -> Agent:
    return Agent(
        name="Trader",
        instructions=prompt,
        tools=tools,
        output_type=AgentAction,
        model=model
    )
