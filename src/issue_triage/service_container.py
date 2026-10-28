"""Composition root for runtime services."""

import logging
from dataclasses import dataclass

from issue_triage import __version__
from issue_triage.config import Settings
from issue_triage.engine import create_engine
from issue_triage.graph import IssueTriageService
from issue_triage.storage import ChromaIssueIndex, IssueRepository


@dataclass(slots=True)
class ServiceContainer:
    settings: Settings
    repository: IssueRepository
    index: ChromaIssueIndex
    triage_service: IssueTriageService
    llm_mode: str
    embedding_mode: str
    version: str = __version__


def build_container(settings: Settings) -> ServiceContainer:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    repository = IssueRepository(settings.database_path)
    index = ChromaIssueIndex(settings)
    engine = create_engine(settings)
    triage_service = IssueTriageService(settings, repository, index, engine)
    return ServiceContainer(
        settings=settings,
        repository=repository,
        index=index,
        triage_service=triage_service,
        llm_mode=engine.mode,
        embedding_mode=index.mode,
    )
