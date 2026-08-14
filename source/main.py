from .prompts import *
from .models import *
from .tools import search_tool, calculator_tool, get_current_datetime

import os
import json
import deepagents
from pathlib import Path
from deepagents.backends import StateBackend
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage
from langchain.tools import tool, tool_node
from functools import wraps
from datetime import datetime
from zoneinfo import ZoneInfo

MAX_ITERATIONS = 3
openai_model = "openai:gpt-4o"
TIMEZONE = "Europe/Kyiv"
load_dotenv()

researcher_agent = deepagents.create_deep_agent(
    model=openai_model,
    system_prompt=RESEARCHER_SYSTEM_PROMPT,
    response_format=ResearchResult,
    tools=[search_tool, calculator_tool, get_current_datetime]
)

planner_llm = ChatOpenAI(
    model="gpt-4o"
).with_structured_output(ResearchPlan)

analyzer_llm = ChatOpenAI(
    model="gpt-4o"
).with_structured_output(AnalysisResult)

checker_llm = ChatOpenAI(
    model="gpt-4o",
).with_structured_output(CheckResult)

synthesis_llm = ChatOpenAI(
    model="gpt-4o"
).with_structured_output(SynthesizedAnswer)


def node_print(func):
    @wraps(func)
    def wrapper(state):
        print(f"Entering '{func.__name__}' node")
        result = func(state)
        print(f"Exit '{func.__name__}' node")
        return result

    return wrapper

def save_graph_image(graph):
    graph.get_graph().draw_mermaid_png(
        output_file_path="graph.png"
    )
    print("Graph saved to graph.png")

@node_print
def current_time_node(state : ResearchState):
    return {"current_time": datetime.now(ZoneInfo(TIMEZONE)).isoformat()}

@node_print
def planner_node(state : ResearchState):
    plan = planner_llm.invoke(build_planner_prompt(state["user_query"], state["current_time"]))
    return { "plan" : plan }
    
@node_print
def researcher_node(state : ResearchState):    
    agent_result = researcher_agent.invoke({"messages" : [HumanMessage(build_researcher_prompt(state))]})
    
    print(agent_result.keys())

    research = agent_result.get("structured_response")

    if research is None:
        raise ValueError(f"Agent returned no structured_response. Keys: {list(agent_result.keys())}")

    return {"research": research}
    
@node_print
def analyzer_node(state : ResearchState):
    analysis = analyzer_llm.invoke(
        ANALYZER_PROMPT.format(
            user_query=state["user_query"],
            plan=state["plan"],
            research=state["research"],
            previous_check=state.get("check_result"),
            date=date_block(state.get("current_time")),
        )
    )
    
    return {
        "analysis" : analysis
    }

@node_print
def checker_node(state : ResearchState):
    check_result = checker_llm.invoke(
            CHECKER_PROMPT.format(
                user_query=state["user_query"],
                plan=state["plan"],
                research=state["research"],
                analysis=state["analysis"],
                iteration_count=state["iteration_count"],
                date=date_block(state.get("current_time")),
        )
    )

    return {
        "check_result": check_result,
        "iteration_count": state["iteration_count"] + 1,
    }
    
@node_print
def synthesizer_node(state : ResearchState):
    synthesis_answer = synthesis_llm.invoke(
        SYNTHESIZER_PROMPT.format(
            user_query=state["user_query"],
            plan=state["plan"],
            research=state["research"],
            analysis=state["analysis"],
            check_result=state["check_result"],
        )
    )

    return {
        "final_answer" : synthesis_answer,
    }

@node_print
def save_result_node(state: ResearchState):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = Path("results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    final_result = state["final_answer"]

    print("Save final answer as JSON")
    with open(output_dir / "research_result.json", "w", encoding="utf-8") as f:
        json.dump(
            final_result.model_dump(),
            f,
            ensure_ascii=False,
            indent=2)

    print("Save final answer as Markdown")
    
    with open(output_dir / "research_result.md", "w", encoding="utf-8") as f:
        f.write(final_result.answer)

    run_result = {
        "user_query": state["user_query"],
        "iteration_count": state["iteration_count"],
        "plan": state["plan"].model_dump() if state["plan"] else None,
        "research": state["research"].model_dump() if state["research"] else None,
        "analysis": state["analysis"].model_dump() if state["analysis"] else None,
        "check_result": (
            state["check_result"].model_dump()
            if state["check_result"]
            else None
        ),
        "final_answer": (
            state["final_answer"].model_dump()
            if state["final_answer"]
            else None
        ),
    }

    print("Save complete pipeline state")
    with open(
        output_dir / "run.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            run_result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Research result saved to {output_dir}")
    return {}

def decide_next_by_checker_result(state: ResearchState):
    if state["iteration_count"] >= MAX_ITERATIONS:
        print("Maximum research iterations reached")
        return "pass"

    decision = state["check_result"].decision

    if decision:
        return decision

    print("Decision result is empty")
    return "research"

langfuse_handler = CallbackHandler()
print("1. langfuse handler created")

graph = StateGraph(ResearchState)
print("2. graph created")

graph.add_node("current_time", current_time_node)
graph.add_node("planner", planner_node)
graph.add_node("research", researcher_node)
graph.add_node("analyzer", analyzer_node)
graph.add_node("checker", checker_node)
graph.add_node("synthesizer", synthesizer_node)
graph.add_node("save_result", save_result_node)

graph.add_edge(START, "current_time")
graph.add_edge("current_time", "planner")
graph.add_edge("planner", "research")
graph.add_edge("research", "analyzer")
graph.add_edge("analyzer", "checker")
graph.add_conditional_edges("checker", decide_next_by_checker_result, 
                            {
                                "pass" : "synthesizer",
                                "research" : "research",
                                "analysis" : "analyzer"
                            })
graph.add_edge("synthesizer", "save_result")
graph.add_edge("save_result", END)

graph = graph.compile()

save_graph_image(graph)

result = graph.invoke(ResearchState(
    user_query="Research possible bitcoin fluctuations and market moves in meantime. Give an answer in which side is better to invest right now",
    current_time=None,
    iteration_count=0,
    plan=None,
    research=None,
    analysis=None,
    check_result=None,
    final_answer=None,
))