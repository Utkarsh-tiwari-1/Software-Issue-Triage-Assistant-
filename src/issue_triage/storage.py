"""SQLite issue history and ChromaDB semantic retrieval."""

import hashlib
import math
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from issue_triage.config import Settings
from issue_triage.schemas import (
    IssueCategory,
    IssuePriority,
    RelatedIssue,
    TriageRequest,
    TriageResponse,
)


class HashEmbeddings(Embeddings):
    """Deterministic, dependency-free feature embeddings for local development."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[\w]+", text.casefold())
        features = list(tokens)
        features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
        for token in tokens:
            features.extend(token[index : index + 3] for index in range(max(1, len(token) - 2)))

        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += 1.0 if digest[4] & 1 else -1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class IssueRepository:
    """Transactional SQLite persistence for complete triage responses."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS issues (
                    issue_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_issues_created_at ON issues(created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_issues_category_priority "
                "ON issues(category, priority)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, request: TriageRequest, response: TriageResponse) -> None:
        with self._session() as connection:
            connection.execute(
                """
                INSERT INTO issues (
                    issue_id, title, description, category, priority, created_at, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(response.issue_id),
                    response.title,
                    request.searchable_text(),
                    response.category.value,
                    response.priority.value,
                    response.created_at.isoformat(),
                    response.model_dump_json(),
                ),
            )

    def get(self, issue_id: UUID) -> TriageResponse | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT response_json FROM issues WHERE issue_id = ?", (str(issue_id),)
            ).fetchone()
        if row is None:
            return None
        return TriageResponse.model_validate_json(row["response_json"])

    def delete(self, issue_id: UUID) -> None:
        with self._session() as connection:
            connection.execute("DELETE FROM issues WHERE issue_id = ?", (str(issue_id),))

    def list(
        self,
        *,
        limit: int,
        offset: int,
        category: IssueCategory | None = None,
        priority: IssuePriority | None = None,
    ) -> tuple[list[TriageResponse], int]:
        conditions: list[str] = []
        parameters: list[Any] = []
        if category is not None:
            conditions.append("category = ?")
            parameters.append(category.value)
        if priority is not None:
            conditions.append("priority = ?")
            parameters.append(priority.value)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._session() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM issues {where_clause}", parameters
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT response_json FROM issues {where_clause} "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            ).fetchall()
        return [TriageResponse.model_validate_json(row["response_json"]) for row in rows], total

    def is_ready(self) -> bool:
        with self._session() as connection:
            connection.execute("SELECT 1").fetchone()
        return True


class ChromaIssueIndex:
    """Persistent vector index with a local/offline embedding option."""

    def __init__(self, settings: Settings) -> None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        if settings.use_openai_embeddings:
            from langchain_openai import OpenAIEmbeddings

            assert settings.openai_api_key is not None
            embeddings: Embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key.get_secret_value(),
                request_timeout=settings.openai_timeout_seconds,
            )
            self.mode = "openai"
        else:
            embeddings = HashEmbeddings()
            self.mode = "local-hash"

        self._store = Chroma(
            collection_name="software_issues",
            persist_directory=str(settings.chroma_path),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, limit: int) -> list[RelatedIssue]:
        matches = self._store.similarity_search_with_score(query, k=limit)
        results: list[RelatedIssue] = []
        for document, distance in matches:
            metadata = document.metadata
            similarity = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
            results.append(
                RelatedIssue(
                    issue_id=UUID(str(metadata["issue_id"])),
                    title=str(metadata["title"]),
                    summary=str(metadata["summary"]),
                    category=IssueCategory(str(metadata["category"])),
                    priority=IssuePriority(str(metadata["priority"])),
                    similarity=similarity,
                )
            )
        return results

    def add(self, request: TriageRequest, response: TriageResponse) -> None:
        content = request.searchable_text()
        metadata = {
            "issue_id": str(response.issue_id),
            "title": response.title,
            "summary": response.summary,
            "category": response.category.value,
            "priority": response.priority.value,
            "created_at": response.created_at.isoformat(),
        }
        self._store.add_documents(
            [Document(page_content=content, metadata=metadata)],
            ids=[str(response.issue_id)],
        )

    def delete(self, issue_id: UUID) -> None:
        self._store.delete(ids=[str(issue_id)])

    def is_ready(self) -> bool:
        self._store._collection.count()
        return True
