import deepagents
from langchain_openai import ChatOpenAI

from .config import DEEP_AGENT_MODEL, GPT4_O_MODEL
from .models import ResearchResult
from .prompts import RESEARCHER_SYSTEM_PROMPT
from .tools import calculator_tool, get_current_datetime, search_tool


def build_structured_llm(schema, model: str = GPT4_O_MODEL):
    return ChatOpenAI(model=model).with_structured_output(schema)


def build_researcher_agent(model: str = DEEP_AGENT_MODEL):
    return deepagents.create_deep_agent(
        model=model,
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        response_format=ResearchResult,
        tools=[search_tool, calculator_tool, get_current_datetime],
    )
