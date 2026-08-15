import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage

from .guardrails import Guardrail, GuardrailViolation
from .config import RESULTS_DIR, TIMEZONE
from .logging_config import get_logger
from .models import ResearchState, GuardrailViolationInfo
from .prompts import (
    ANALYZER_PROMPT,
    CHECKER_PROMPT,
    SYNTHESIZER_PROMPT,
    build_planner_prompt,
    build_researcher_prompt,
    date_block,
)


def _violation_info(node: str, error: GuardrailViolation) -> GuardrailViolationInfo:
    return GuardrailViolationInfo(
        guardrail=error.exception_name,
        node=error.node or node,
        attempt=error.attempt,
        message=str(error),
    )


class Node(ABC):
    name: str
    pre_checks: tuple[Guardrail] = ()
    post_checks: tuple[Guardrail] = ()

    def __call__(self, state: ResearchState) -> dict:
        print(f"Entering '{self.name}' node")
        get_logger().info("node=%s", self.name)
        try:
            for check in self.pre_checks:
                check(state)
            result = self.run(state)
            for check in self.post_checks:
                check(state, result)
        except GuardrailViolation as error:
            get_logger().warning(
                "Guardrail triggered | node=%s | guardrail=%s | attempt=%s | message=%s",
                self.name, error.exception_name, error.attempt, str(error),
            )
            print(f"Guardrail '{error.exception_name}' triggered in '{self.name}' node: {error}")
            return {"guardrail_violation": _violation_info(self.name, error)}

        get_logger().info("node=%s updated=%s", self.name, list(result))
        print(f"Exit '{self.name}' node")
        return result

    @abstractmethod
    def run(self, state: ResearchState) -> dict:
        ...


class GuardrailNode(Node):
    def __init__(self, name: str, guardrails: list[Guardrail]):
        self.name = name
        self.pre_checks = tuple(guardrails)

    def run(self, state: ResearchState) -> dict:
        return {}


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

    def __init__(self, agent, pre_checks: tuple[Guardrail] = ()):
        self.agent = agent
        self.pre_checks = tuple(pre_checks)

    def run(self, state: ResearchState) -> dict:
        try:
            agent_result = self.agent.invoke(
                {"messages": [HumanMessage(build_researcher_prompt(state))]}
            )
        except Exception as error:
            raise GuardrailViolation(
                message=f"Research agent failed: {error}",
                node="research",
                attempt=state["iteration_count"],
                exception_name="researcher_response_guardrail",
            )

        research = agent_result.get("structured_response")

        if research is None:
            raise GuardrailViolation(
                message="Agent returned no structured_response",
                node="research",
                attempt=state["iteration_count"],
                exception_name="researcher_response_guardrail",
            )

        return {"research": research}


class AnalyzerNode(Node):
    name = "analyzer"

    def __init__(self, llm, pre_checks: tuple[Guardrail] = ()):
        self.llm = llm
        self.pre_checks = tuple(pre_checks)

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

    def __init__(self, llm, iteration_guardrail: Guardrail | None = None):
        self.llm = llm
        self.iteration_guardrail = iteration_guardrail

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

        new_count = state["iteration_count"] + 1
        result = {"check_result": check_result, "iteration_count": new_count}

        if self.iteration_guardrail is not None:
            try:
                self.iteration_guardrail.check({**state, "iteration_count": new_count})
            except GuardrailViolation as error:
                get_logger().warning(
                    "Iteration guardrail reached | attempt=%s | message=%s",
                    error.attempt, str(error),
                )
                print(f"Iteration limit reached: {error}. Synthesizing from current state.")
                result["guardrail_violation"] = _violation_info(self.name, error)

        return result


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

        violation = state.get("guardrail_violation")
        final_result = state.get("final_answer")

        if final_result is not None:
            print("Save final answer as JSON")
            with open(output_dir / "research_result.json", "w", encoding="utf-8") as f:
                json.dump(final_result.model_dump(), f, ensure_ascii=False, indent=2)

            print("Save final answer as Markdown")
            with open(output_dir / "research_result.md", "w", encoding="utf-8") as f:
                f.write(final_result.answer)

        if violation is not None:
            print("Save guardrail error")
            with open(output_dir / "guardrail_error.json", "w", encoding="utf-8") as f:
                json.dump(violation.model_dump(), f, ensure_ascii=False, indent=2)
            get_logger().warning(
                "Pipeline result saved with guardrail error | guardrail=%s | node=%s | attempt=%s | message=%s",
                violation.guardrail, violation.node, violation.attempt, violation.message,
            )

        run_result = {
            "user_query": state["user_query"],
            "iteration_count": state["iteration_count"],
            "guardrail_violation": violation.model_dump() if violation else None,
            "plan": state["plan"].model_dump() if state.get("plan") else None,
            "research": state["research"].model_dump() if state.get("research") else None,
            "analysis": state["analysis"].model_dump() if state.get("analysis") else None,
            "check_result": (
                state["check_result"].model_dump() if state.get("check_result") else None
            ),
            "final_answer": final_result.model_dump() if final_result else None,
        }

        print("Save complete pipeline state")
        with open(output_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_result, f, ensure_ascii=False, indent=2)

        # Console output.
        if violation is not None:
            print(f"\n[GUARDRAIL ERROR] {violation.guardrail} @ {violation.node}: {violation.message}")
        if final_result is not None:
            print("\n=== FINAL ANSWER ===")
            print(final_result.answer)
        elif violation is not None:
            print("Pipeline stopped by guardrail — no final answer was produced.")

        print(f"\nResearch result saved to {output_dir}")
        return {}

class CheckerRouter:
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __call__(self, state: ResearchState) -> str:
        if state["iteration_count"] >= self.max_iterations:
            print("Maximum research iterations reached")
            get_logger().info("router=checker action=pass")
            return "pass"

        decision = state["check_result"].decision

        if not decision:
            print("Decision result is empty")
            decision = "research"

        get_logger().info("router=checker action=%s", decision)
        return decision


class GuardrailRouter:
    def __call__(self, state: ResearchState) -> str:
        action = "blocked" if state.get("guardrail_violation") else "continue"
        get_logger().info("router=guardrail action=%s", action)
        return action
