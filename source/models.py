from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

max_research_tasks_count = 3

# Research

class ResearchTask(BaseModel):
    question: str = Field(description="Specific question that must be answered.")
    purpose: str = Field(description="Why this information is needed for the research.")

class ResearchPlan(BaseModel):
    tasks : list[ResearchTask] = Field(default_factory=list, max_length=max_research_tasks_count)
    require_current_time : bool = False

class SearchResult(BaseModel):
    title : str
    url : str
    content : str
    published_date: str | None = None
    
class ResearchSource(BaseModel):
    id : str
    title : str
    url : str
    content : str
    published_date: str | None = None
    
class ResearchFinding(BaseModel):
    id: str
    statement: str
    supporting_source_ids: list[str] = Field(default_factory=list, description="IDs of sources supporting this finding.")

class ResearchResult(BaseModel):
    findings : list[ResearchFinding]
    sources : list[ResearchSource]

    def add_finding(self, finding : str, source : ResearchSource):
        self.findings.append(finding)
        self.sources.append(source)

# Analysis

class AnalysisClaim(BaseModel):
    claim: str
    supporting_source_ids: list[str] = Field(default_factory=list, description="URLs or source identifiers supporting this claim.")
    confidence: float = Field(ge=0.0, le=1.0)

class AnalysisResult(BaseModel):
    summary: str
    claims: list[AnalysisClaim]
    contradictions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)

# Checker

class CheckDecision(str, Enum):
    PASS = "pass"
    RESEARCH = "research"
    ANALYSIS = "analysis"

class CheckResult(BaseModel):
    decision: CheckDecision
    reasoning: str
    missing_information: list[str] = Field(default_factory=list)
    incorrect_claims: list[str] = Field(default_factory=list)
    additional_queries: list[str] = Field(default_factory=list)
    
# Synthesizer

class SynthesizedAnswer(BaseModel):
    answer: str
    supporting_source_ids : list[str] = Field(default_factory=list)

# LangGraph state    
    
class ResearchState(TypedDict):
    user_query: str
    current_time : str | None
    iteration_count: int
    plan: ResearchPlan | None
    research: ResearchResult | None
    analysis: AnalysisResult | None
    check_result: CheckResult | None
    final_answer: SynthesizedAnswer | None