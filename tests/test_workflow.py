from fastapi.testclient import TestClient


def test_exact_repeat_routes_to_duplicate_branch(client: TestClient) -> None:
    report = {
        "title": "Database connection pool exhausted",
        "description": "The API fails with timeout because the database connection pool is exhausted.",
        "environment": "production",
        "language": "Python",
    }

    first = client.post("/api/v1/triage", json=report)
    second = client.post("/api/v1/triage", json=report)

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["duplicate_of"] == first.json()["issue_id"]
    assert second.json()["related_issues"][0]["similarity"] >= 0.99


def test_security_and_outage_signals_get_elevated_priority(client: TestClient) -> None:
    security = client.post(
        "/api/v1/triage",
        json={
            "title": "Authentication bypass",
            "description": "A security vulnerability allows an auth bypass in production.",
        },
    )
    outage = client.post(
        "/api/v1/triage",
        json={
            "title": "Production outage",
            "description": "Production is down and all users receive service unavailable.",
        },
    )

    assert security.json()["category"] == "security"
    assert security.json()["priority"] == "P1"
    assert outage.json()["priority"] == "P0"
