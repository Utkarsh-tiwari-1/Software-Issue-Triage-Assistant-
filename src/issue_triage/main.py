"""FastAPI application factory and ASGI entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from issue_triage import __version__
from issue_triage.api import RequestIdMiddleware, router, validation_exception_handler
from issue_triage.config import Settings, get_settings
from issue_triage.service_container import build_container


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.container = build_container(runtime_settings)
        yield

    app = FastAPI(
        title=runtime_settings.app_name,
        version=__version__,
        description=(
            "A LangGraph-powered service that classifies software issues, "
            "retrieves similar reports, and recommends debugging steps."
        ),
        docs_url="/docs" if runtime_settings.environment != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )
    app.include_router(router)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": runtime_settings.app_name, "docs": "/docs"}

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("issue_triage.api").exception(
            "Unhandled application error",
            extra={"request_id": getattr(request.state, "request_id", "unknown")},
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    return app


app = create_app()
