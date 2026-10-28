from fastapi.testclient import TestClient

from issue_triage.config import Settings
from issue_triage.main import create_app


def sample_report() -> dict[str, str]:
    return {
        "title": "Checkout fails with saved cards",
        "description": (
            "Checkout returns a TypeError when a customer has a saved card after the SDK upgrade."
        ),
        "environment": "production API v2",
        "language": "Python",
        "repository": "payments/api",
    }


def test_triage_persists_and_returns_typed_analysis(client: TestClient) -> None:
    response = client.post("/api/v1/triage", json=sample_report())

    assert response.status_code == 201
    payload = response.json()
    assert payload["category"] == "bug"
    assert payload["priority"] == "P2"
    assert payload["debugging_steps"]
    assert payload["analysis_mode"] == "local-rules"
    assert response.headers["x-request-id"]

    fetched = client.get(f"/api/v1/issues/{payload['issue_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["issue_id"] == payload["issue_id"]


def test_validation_errors_do_not_echo_submitted_values(client: TestClient) -> None:
    response = client.post(
        "/api/v1/triage",
        json={"title": "private-value", "description": "   ", "unknown": "secret"},
    )

    assert response.status_code == 422
    assert "private-value" not in response.text
    assert "secret" not in response.text
    assert response.json()["request_id"]


def test_issue_list_filters_by_category_and_priority(client: TestClient) -> None:
    client.post("/api/v1/triage", json=sample_report())
    client.post(
        "/api/v1/triage",
        json={
            "title": "Production database is down",
            "description": "Production is down: every request fails with service unavailable.",
        },
    )

    response = client.get("/api/v1/issues?category=bug&priority=P2&limit=10")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["priority"] == "P2"


def test_api_key_protects_issue_routes(settings: Settings) -> None:
    protected_settings = Settings.model_validate(
        {**settings.model_dump(), "api_key": "test-secret"}
    )
    with TestClient(create_app(protected_settings)) as client:
        denied = client.post("/api/v1/triage", json=sample_report())
        allowed = client.post(
            "/api/v1/triage", json=sample_report(), headers={"X-API-Key": "test-secret"}
        )

    assert denied.status_code == 401
    assert allowed.status_code == 201
