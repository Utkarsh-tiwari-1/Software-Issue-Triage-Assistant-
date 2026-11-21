"""Public API models and validated internal workflow outputs."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class IssueCategory(StrEnum):
    BUG = "bug"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USABILITY = "usability"
    DOCUMENTATION = "documentation"
    BUILD = "build"
    DEPLOYMENT = "deployment"
    QUESTION = "question"
    UNKNOWN = "unknown"


class IssuePriority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TriageRequest(BaseModel):
    """A report submitted by a user for triage."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, max_length=200)
    description: Annotated[NonEmptyText, StringConstraints(max_length=20_000)]
    environment: str | None = Field(default=None, max_length=1_000)
    language: str | None = Field(default=None, max_length=100)
    stack_trace: str | None = Field(default=None, max_length=30_000)
    repository: str | None = Field(default=None, max_length=300)

    def searchable_text(self) -> str:
        """Build the text used for retrieval and analysis."""

        fields = [
            self.title,
            self.description,
            self.environment,
            self.language,
            self.stack_trace,
            self.repository,
        ]
        return "\n".join(value for value in fields if value)


class TriageAnalysis(BaseModel):
    """Schema-enforced classification output from the analysis stage."""

    model_config = ConfigDict(extra="forbid")

    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=800)]
    category: IssueCategory
    priority: IssuePriority
    confidence: float = Field(ge=0, le=1)
    rationale: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=10, max_length=1_500)
    ]
    tags: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
    ] = Field(default_factory=list, max_length=8)


class RecommendationPlan(BaseModel):
    """Schema-enforced debugging guidance output."""

    model_config = ConfigDict(extra="forbid")

    debugging_steps: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=8, max_length=500)]
    ] = Field(min_length=2, max_length=8)


class RelatedIssue(BaseModel):
    issue_id: UUID
    title: str
    summary: str
    category: IssueCategory
    priority: IssuePriority
    similarity: float = Field(ge=0, le=1)


class TriageResponse(BaseModel):
    issue_id: UUID
    title: str
    summary: str
    category: IssueCategory
    priority: IssuePriority
    confidence: float = Field(ge=0, le=1)
    rationale: str
    debugging_steps: list[str]
    tags: list[str]
    related_issues: list[RelatedIssue]
    duplicate_of: UUID | None = None
    created_at: datetime
    analysis_mode: str


class IssueListResponse(BaseModel):
    items: list[TriageResponse]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_mode: str
    embedding_mode: str
