"""LangGraph workflow for validation, retrieval, tool use, and response generation."""

import json
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict
from uuid import UUID, uuid4

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from issue_triage.config import Settings
from issue_triage.engine import LocalTriageEngine, OpenAITriageEngine
from issue_triage.playbooks import get_playbook
from issue_triage.schemas import (
    IssueCategory,
    IssuePriority,
    RelatedIssue,
    TriageAnalysis,
    TriageRequest,
    TriageResponse,
)
from issue_triage.storage import ChromaIssueIndex, IssueRepository


class TriageState(TypedDict, total=False):
    request: TriageRequest
    issue_id: UUID
    related_issues: list[RelatedIssue]
    messages: Annotated[list[AnyMessage], add_messages]
    tool_context: list[str]
    analysis: TriageAnalysis
    debugging_steps: list[str]
    duplicate_of: UUID | None
    response: TriageResponse
    validation_error: str


class IssueTriageService:
    """Build and execute the issue triage graph over persistent stores."""

    def __init__(
        self,
        settings: Settings,
        repository: IssueRepository,
        index: ChromaIssueIndex,
        engine: LocalTriageEngine | OpenAITriageEngine,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.index = index
        self.engine = engine
        self.graph = self._build_graph()

    def triage(self, request: TriageRequest) -> TriageResponse:
        final_state = self.graph.invoke(
            {"request": request, "issue_id": uuid4(), "tool_context": []},
            config={"recursion_limit": 12},
        )
        return final_state["response"]

    def get_issue(self, issue_id: UUID) -> TriageResponse | None:
        return self.repository.get(issue_id)

    def list_issues(
        self,
        *,
        limit: int,
        offset: int,
        category: IssueCategory | None = None,
        priority: IssuePriority | None = None,
    ) -> tuple[list[TriageResponse], int]:
        return self.repository.list(limit=limit, offset=offset, category=category, priority=priority)

    def _build_graph(self) -> Any:
        @tool
        def search_known_issues(query: str) -> str:
            """Search previously triaged reports for similar known issues."""
            matches = self.index.search(query, self.settings.retrieval_k)
            return json.dumps([item.model_dump(mode="json") for item in matches])

        @tool
        def get_debugging_playbook(category: str) -> str:
            """Retrieve the standard first-response debugging checklist for a category."""
            try:
                parsed_category = IssueCategory(category.casefold())
            except ValueError:
                parsed_category = IssueCategory.UNKNOWN
            return json.dumps(get_playbook(parsed_category))

        tools = [search_known_issues, get_debugging_playbook]
        tool_node = ToolNode(tools, handle_tool_errors=True)
        workflow = StateGraph(TriageState)

        def validate(state: TriageState) -> dict[str, Any]:
            request = TriageRequest.model_validate(state["request"])
            if not request.description.strip():
                raise ValueError("description cannot be empty")
            return {"request": request}

        def retrieve(state: TriageState) -> dict[str, Any]:
            matches = self.index.search(state["request"].searchable_text(), self.settings.retrieval_k)
            return {"related_issues": matches}

        def should_gather_context(state: TriageState) -> str:
            return "context_agent" if isinstance(self.engine, OpenAITriageEngine) else "analyze"

        def gather_context(state: TriageState) -> dict[str, Any]:
            message = self.engine.gather_context(state["request"], tools)
            return {"messages": [message]}

        def route_tool_calls(state: TriageState) -> str:
            last_message = state["messages"][-1]
            return "tools" if getattr(last_message, "tool_calls", None) else "analyze"

        def analyze(state: TriageState) -> dict[str, Any]:
            context = self._tool_context(state.get("messages", []))
            analysis = self.engine.analyze(
                state["request"], state.get("related_issues", []), context
            )
            return {"analysis": analysis, "tool_context": context}

        def route_duplicate(state: TriageState) -> str:
            related = state.get("related_issues", [])
            if related and related[0].similarity >= self.settings.duplicate_threshold:
                return "duplicate"
            return "new_issue"

        def mark_duplicate(state: TriageState) -> dict[str, Any]:
            return {"duplicate_of": state["related_issues"][0].issue_id}

        def mark_new_issue(state: TriageState) -> dict[str, Any]:
            return {"duplicate_of": None}

        def recommend(state: TriageState) -> dict[str, Any]:
            plan = self.engine.recommend(
                state["request"],
                state["analysis"],
                state.get("related_issues", []),
                state.get("tool_context", []),
            )
            return {"debugging_steps": plan.debugging_steps}

        def persist(state: TriageState) -> dict[str, Any]:
            request = state["request"]
            analysis = state["analysis"]
            response = TriageResponse(
                issue_id=state["issue_id"],
                title=request.title or self._title_from_description(request.description),
                summary=analysis.summary,
                category=analysis.category,
                priority=analysis.priority,
                confidence=analysis.confidence,
                rationale=analysis.rationale,
                debugging_steps=state["debugging_steps"],
                tags=analysis.tags,
                related_issues=state.get("related_issues", []),
                duplicate_of=state.get("duplicate_of"),
                created_at=datetime.now(UTC),
                analysis_mode=self.engine.mode,
            )
            self.repository.save(request, response)
            try:
                self.index.add(request, response)
            except Exception:
                try:
                    self.index.delete(response.issue_id)
                finally:
                    self.repository.delete(response.issue_id)
                raise
            return {"response": response}

        workflow.add_node("validate", validate)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("context_agent", gather_context)
        workflow.add_node("tools", tool_node)
        workflow.add_node("analyze", analyze)
        workflow.add_node("mark_duplicate", mark_duplicate)
        workflow.add_node("mark_new_issue", mark_new_issue)
        workflow.add_node("recommend", recommend)
        workflow.add_node("persist", persist)

        workflow.add_edge(START, "validate")
        workflow.add_edge("validate", "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            should_gather_context,
            {"context_agent": "context_agent", "analyze": "analyze"},
        )
        workflow.add_conditional_edges(
            "context_agent", route_tool_calls, {"tools": "tools", "analyze": "analyze"}
        )
        workflow.add_edge("tools", "analyze")
        workflow.add_conditional_edges(
            "analyze",
            route_duplicate,
            {"duplicate": "mark_duplicate", "new_issue": "mark_new_issue"},
        )
        workflow.add_edge("mark_duplicate", "recommend")
        workflow.add_edge("mark_new_issue", "recommend")
        workflow.add_edge("recommend", "persist")
        workflow.add_edge("persist", END)
        return workflow.compile()

    @staticmethod
    def _tool_context(messages: list[AnyMessage]) -> list[str]:
        results: list[str] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                content = message.content
                if isinstance(content, str):
                    results.append(content[:8_000])
        return results

    @staticmethod
    def _title_from_description(description: str) -> str:
        title = " ".join(description.split())
        return title[:197] + "..." if len(title) > 200 else title
