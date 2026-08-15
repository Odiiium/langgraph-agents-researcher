import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage

from .guardrails import *
from .config import RESULTS_DIR, TIMEZONE
from .models import ResearchState
from .prompts import (
    ANALYZER_PROMPT,
    CHECKER_PROMPT,
    SYNTHESIZER_PROMPT,
    build_planner_prompt,
    build_researcher_prompt,
    date_block,
)


class Node(ABC):
    name: str
    pre_checks : tuple[Guardrail] = ()
    post_checks : tuple[Guardrail] = ()

    def __call__(self, state: ResearchState) -> dict:
        print(f"Entering '{self.name}' node")
        for check in self.pre_checks:
            check(state)
        result = self.run(state)
        for check in self.post_checks:
            check(state, result)
        print(f"Exit '{self.name}' node")
        return result

    @abstractmethod
    def run(self, state: ResearchState) -> dict:
        ...


class CurrentTimeNode(Node):
    name = "current_time"

    def __init__(self, timezone: str = TIMEZONE):
        self.timezone = timezone

    def run(self, state: ResearchState) -> dict:
        return {"current_time": datetime.now(ZoneInfo(self.timezone)).isoformat()}


class PlannerNode(Node):
    name = "planner"

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: ResearchState) -> dict:
        plan = self.llm.invoke(
            build_planner_prompt(state["user_query"], state["current_time"])
        )
        return {"plan": plan}


class ResearcherNode(Node):
    name = "research"

    def __init__(self, agent, pre_checks = ()):
        self.agent = agent
        self.pre_checks = pre_checks

    def run(self, state: ResearchState) -> dict:

        
        agent_result = self.agent.invoke({"messages": [HumanMessage(build_researcher_prompt(state))]})

        research = agent_result.get("structured_response")

        if research is None:
            raise ValueError(f"Agent returned no structured_response. Keys: {list(agent_result.keys())}")

        return {"research": research}


class AnalyzerNode(Node):
    name = "analyzer"

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: ResearchState) -> dict:
        analysis = self.llm.invoke(
            ANALYZER_PROMPT.format(
                user_query=state["user_query"],
                plan=state["plan"],
                research=state["research"],
                previous_check=state.get("check_result"),
                date=date_block(state.get("current_time")),
            )
        )
        return {"analysis": analysis}


class CheckerNode(Node):
    name = "checker"

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: ResearchState) -> dict:
        check_result = self.llm.invoke(
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


class SynthesizerNode(Node):
    name = "synthesizer"

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: ResearchState) -> dict:
        synthesis_answer = self.llm.invoke(
            SYNTHESIZER_PROMPT.format(
                user_query=state["user_query"],
                plan=state["plan"],
                research=state["research"],
                analysis=state["analysis"],
                check_result=state["check_result"],
            )
        )
        return {"final_answer": synthesis_answer}


class SaveResultNode(Node):
    name = "save_result"

    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir

    def run(self, state: ResearchState) -> dict:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        output_dir = Path(self.results_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        final_result = state["final_answer"]

        print("Save final answer as JSON")
        with open(output_dir / "research_result.json", "w", encoding="utf-8") as f:
            json.dump(final_result.model_dump(), f, ensure_ascii=False, indent=2)

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
                state["check_result"].model_dump() if state["check_result"] else None
            ),
            "final_answer": (
                state["final_answer"].model_dump() if state["final_answer"] else None
            ),
        }

        print("Save complete pipeline state")
        with open(output_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_result, f, ensure_ascii=False, indent=2)

        print(f"Research result saved to {output_dir}")
        return {}


class CheckerRouter:
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __call__(self, state: ResearchState) -> str:
        if state["iteration_count"] >= self.max_iterations:
            print("Maximum research iterations reached")
            return "pass"

        decision = state["check_result"].decision

        if decision:
            return decision

        print("Decision result is empty")
        return "research"

class GuardrailRouter:
    def __call__(self, state: ResearchState) -> str:
        if state.get("guardrail_violation"):
            return "blocked"

        return "continue"