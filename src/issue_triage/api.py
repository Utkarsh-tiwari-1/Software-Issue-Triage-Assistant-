"""HTTP API routes, authentication, request IDs, and exception handling."""

import hmac
import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import RequestResponseEndpoint

from issue_triage.schemas import (
    HealthResponse,
    IssueCategory,
    IssueListResponse,
    IssuePriority,
    TriageRequest,
    TriageResponse,
)
from issue_triage.service_container import ServiceContainer

logger = logging.getLogger("issue_triage.api")
router = APIRouter()


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def require_api_key(
    container: ServiceContainer = Depends(get_container),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    configured_key = container.settings.api_key
    if configured_key is None:
        return
    expected = configured_key.get_secret_value()
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.get("/health/live", response_model=HealthResponse, tags=["health"])
def liveness(container: ServiceContainer = Depends(get_container)) -> HealthResponse:
    return HealthResponse(
        status="alive",
        version=container.version,
        llm_mode=container.llm_mode,
        embedding_mode=container.embedding_mode,
    )


@router.get("/health/ready", response_model=HealthResponse, tags=["health"])
def readiness(container: ServiceContainer = Depends(get_container)) -> HealthResponse:
    try:
        container.repository.is_ready()
        container.index.is_ready()
    except Exception as exc:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail="A backing service is not ready") from exc
    return HealthResponse(
        status="ready",
        version=container.version,
        llm_mode=container.llm_mode,
        embedding_mode=container.embedding_mode,
    )


@router.post(
    "/api/v1/triage",
    response_model=TriageResponse,
    status_code=201,
    tags=["triage"],
    dependencies=[Depends(require_api_key)],
)
def triage_issue(
    payload: TriageRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> TriageResponse:
    try:
        result = container.triage_service.triage(payload)
        logger.info(
            "Issue triaged",
            extra={
                "request_id": request.state.request_id,
                "issue_id": str(result.issue_id),
                "category": result.category.value,
                "priority": result.priority.value,
            },
        )
        return result
    except Exception as exc:
        logger.exception("Issue triage failed", extra={"request_id": request.state.request_id})
        raise HTTPException(
            status_code=503,
            detail=f"Triage could not be completed. Reference request {request.state.request_id}.",
        ) from exc


@router.get(
    "/api/v1/issues",
    response_model=IssueListResponse,
    tags=["issues"],
    dependencies=[Depends(require_api_key)],
)
def list_issues(
    container: ServiceContainer = Depends(get_container),
    category: IssueCategory | None = None,
    priority: IssuePriority | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> IssueListResponse:
    items, total = container.triage_service.list_issues(
        limit=limit, offset=offset, category=category, priority=priority
    )
    return IssueListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get(
    "/api/v1/issues/{issue_id}",
    response_model=TriageResponse,
    tags=["issues"],
    dependencies=[Depends(require_api_key)],
)
def get_issue(
    issue_id: UUID,
    container: ServiceContainer = Depends(get_container),
) -> TriageResponse:
    result = container.triage_service.get_issue(issue_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Issue not found")
    return result


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = [
        {
            "loc": error.get("loc", []),
            "msg": error.get("msg", "Invalid value"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": errors, "request_id": request.state.request_id},
    )
