from fastapi.testclient import TestClient

from verilogic_ns_api.config import Settings
from verilogic_ns_api.main import create_app


def make_client() -> TestClient:
    settings = Settings(
        service_name="VeriLogic-NS API",
        version="0.1.0",
        cors_origins=["http://localhost:3000"],
    )
    return TestClient(create_app(settings))


def test_health_endpoint_returns_service_status_and_version() -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "service": "VeriLogic-NS API",
        "status": "ok",
        "version": "0.1.0",
    }


def test_cors_allows_only_configured_local_origin() -> None:
    client = make_client()

    allowed = client.get("/health", headers={"Origin": "http://localhost:3000"})
    denied = client.get("/health", headers={"Origin": "http://untrusted.example"})

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in denied.headers
