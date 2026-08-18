"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.llm.report_generator import ReportGenerator
from src.main import create_app
from src.pipeline.orchestrator import AnalyticsOrchestrator
from tests.fixtures.mocks import (
    InMemoryNeo4jClient,
    MockLLMClient,
    MockTBAClient,
    MockTrackingProvider,
)


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def mock_neo4j() -> InMemoryNeo4jClient:
    client = InMemoryNeo4jClient()
    client.connect()
    return client


@pytest.fixture
def mock_tracking() -> MockTrackingProvider:
    return MockTrackingProvider()


@pytest.fixture
def orchestrator(
    mock_llm: MockLLMClient,
    mock_neo4j: InMemoryNeo4jClient,
    mock_tracking: MockTrackingProvider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> AnalyticsOrchestrator:
    db_path = tmp_path / "test.db"
    orch = AnalyticsOrchestrator(
        database_url=f"sqlite:///{db_path}",
        tba_api_key="test-key",
        neo4j_client=mock_neo4j,
        report_generator=ReportGenerator(llm=mock_llm),
        tracking_provider=mock_tracking,
    )

    async def mock_tba_factory(api_key: str = "test-key"):
        return MockTBAClient(api_key)

    monkeypatch.setattr(
        "src.pipeline.orchestrator.TBAClient",
        lambda api_key=None, **kwargs: MockTBAClient(api_key or "test-key"),
    )
    return orch


@pytest.fixture
def client(orchestrator: AnalyticsOrchestrator) -> TestClient:
    app = create_app(orchestrator=orchestrator)
    return TestClient(app)
