from dataclasses import dataclass
from typing import Callable

from langgraph.graph import END, START, StateGraph

from .config import *
from .guardrails import *
from .llms import build_structured_llm, build_researcher_agent
from .models import (
    AnalysisResult,
    CheckResult,
    InjectionCheck,
    ResearchPlan,
    ResearchState,
    SynthesizedAnswer,
)
from .nodes import (
    AnalyzerNode,
    GuardrailNode,
    CheckerNode,
    CheckerRouter,
    CurrentTimeNode,
    Node,
    PlannerNode,
    ResearcherNode,
    SaveResultNode,
    SynthesizerNode,
    GuardrailRouter
)


@dataclass
class ConditionalEdge:
    source: str
    router: Callable
    mapping: dict

class Pipeline:
    def __init__(
        self,
        state_schema,
        nodes: list[Node],
        edges: list[tuple[str, str]],
        conditional_edges: list[ConditionalEdge] | None = None,
    ):
        self.state_schema = state_schema
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.conditional_edges = list(conditional_edges or [])
        self.compiled = None

    def build(self) -> "Pipeline":
        graph = StateGraph(self.state_schema)

        for node in self.nodes:
            graph.add_node(node.name, node)

        for source, target in self.edges:
            graph.add_edge(source, target)

        for edge in self.conditional_edges:
            graph.add_conditional_edges(edge.source, edge.router, edge.mapping)

        self.compiled = graph.compile()
        return self

    def save_image(self, path: str = GRAPH_IMAGE_PATH) -> None:
        if self.compiled is None:
            raise RuntimeError("Pipeline must be built before saving the graph image.")
        self.compiled.get_graph().draw_mermaid_png(output_file_path=path)
        print(f"Graph saved to {path}")

    def run(self, initial_state, callbacks=None):
        if self.compiled is None:
            raise RuntimeError("Pipeline must be built before running.")
        return self.compiled.invoke(
            initial_state,
            config={"callbacks": callbacks or []},
        )


def build_research_pipeline(max_iterations: int = MAX_ITERATIONS) -> Pipeline:
    """Wire the concrete research nodes and edges into a ``Pipeline``."""

    nodes = [
        CurrentTimeNode(),
        GuardrailNode("pre_planner_guardrail_node", guardrails=[
            EmptyQueryGuardrail(),
            InputLengthGuardrail(),
            PromptInjectionGuardrail(
                llm=build_structured_llm(
                    schema=InjectionCheck,
                    model=GPT4_O_MODEL,
                )
            )
        ]),
        PlannerNode(llm=build_structured_llm(ResearchPlan, model=GPT5_MODEL)),
        ResearcherNode(
            agent=build_researcher_agent(),
            pre_checks=(EmptyPlanGuardrail(),),
        ),
        AnalyzerNode(
            llm=build_structured_llm(AnalysisResult, GPT5_MODEL),
            pre_checks=(ResearchGroundingGuardrail(),),
        ),
        CheckerNode(
            llm=build_structured_llm(CheckResult, O4_MINI_MODEL),
            iteration_guardrail=ResearchIterationGuardrail(),
        ),
        SynthesizerNode(llm=build_structured_llm(SynthesizedAnswer, GPT5_MINI_MODEL)),
        SaveResultNode(),
    ]

    edges = [
        (START, "pre_planner_guardrail_node"),
        ("current_time", "planner"),
        ("planner", "research"),
        ("synthesizer", "save_result"),
        ("save_result", END),
    ]

    conditional_edges = [
        ConditionalEdge(
            source="pre_planner_guardrail_node",
            router=GuardrailRouter(),
            mapping={
                "blocked": "save_result",
                "continue": "current_time",
            },
        ),
        ConditionalEdge(
            source="research",
            router=GuardrailRouter(),
            mapping={
                "blocked": "save_result",
                "continue": "analyzer",
            },
        ),
        ConditionalEdge(
            source="analyzer",
            router=GuardrailRouter(),
            mapping={
                "blocked": "save_result",
                "continue": "checker",
            },
        ),
        ConditionalEdge(
            source="checker",
            router=CheckerRouter(max_iterations),
            mapping={
                "pass": "synthesizer",
                "research": "research",
                "analysis": "analyzer",
            },
        )
    ]

    return Pipeline(
        state_schema=ResearchState,
        nodes=nodes,
        edges=edges,
        conditional_edges=conditional_edges,
    )
