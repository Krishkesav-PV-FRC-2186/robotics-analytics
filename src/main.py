"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.pipeline.orchestrator import AnalyticsOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    orchestrator = AnalyticsOrchestrator()
    app.state.orchestrator = orchestrator
    yield
    orchestrator.close()


def create_app(orchestrator: AnalyticsOrchestrator | None = None) -> FastAPI:
    """Application factory supporting dependency injection for tests."""
    app = FastAPI(
        title="Robotics Analytics Platform",
        description="FRC competition sports analytics API",
        version="1.0.0",
        lifespan=lifespan if orchestrator is None else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if orchestrator is not None:
        app.state.orchestrator = orchestrator

    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
