from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from issue_triage.config import Settings
from issue_triage.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        environment="test",
        llm_provider="local",
        embedding_provider="local",
        database_path=tmp_path / "issues.sqlite3",
        chroma_path=tmp_path / "chroma",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
