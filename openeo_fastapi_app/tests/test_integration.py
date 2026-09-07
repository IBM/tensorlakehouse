"""Integration tests requiring openeo-fastapi (Python 3.10/3.11).

These tests are automatically skipped on Python 3.12+ where openeo-fastapi
is not installable.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from openeo_fastapi_app.tests.conftest import requires_openeo_fastapi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_STAC_COLLECTIONS = {
    "collections": [
        {
            "id": "test-collection",
            "title": "Test Collection",
            "description": "A test collection",
            "license": "proprietary",
            "stac_version": "1.0.0",
            "extent": {
                "spatial": {"bbox": [[-180, -90, 180, 90]]},
                "temporal": {"interval": [["2020-01-01T00:00:00Z", None]]},
            },
            "links": [],
        }
    ],
    "links": [],
}

MOCK_STAC_COLLECTION = MOCK_STAC_COLLECTIONS["collections"][0]

MOCK_STAC_ITEMS = {
    "type": "FeatureCollection",
    "features": [],
    "links": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_client():
    """Create a FastAPI TestClient for the OpenEO app."""
    from fastapi.testclient import TestClient
    from openeo_fastapi_app.app import create_app

    application = create_app()
    with TestClient(application) as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_openeo_fastapi
class TestCapabilities:
    def test_well_known(self, test_client):
        resp = test_client.get("/.well-known/openeo")
        assert resp.status_code == 200
        data = resp.json()
        assert "versions" in data

    def test_capabilities(self, test_client):
        resp = test_client.get("/openeo/1.1.0/")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("api_version") == "1.1.0"
        assert "endpoints" in data

    def test_health(self, test_client):
        resp = test_client.get("/openeo/1.1.0/health")
        assert resp.status_code == 200

    def test_conformance(self, test_client):
        resp = test_client.get("/openeo/1.1.0/conformance")
        assert resp.status_code == 200

    def test_file_formats(self, test_client):
        resp = test_client.get("/openeo/1.1.0/file_formats")
        assert resp.status_code == 200
        data = resp.json()
        assert "input" in data
        assert "output" in data
        assert "GTiff" in data["output"]
        assert "netCDF" in data["output"]


@requires_openeo_fastapi
class TestCollections:
    def test_get_collections_proxied(self, test_client):
        with patch(
            "openeo_fastapi_app.collections."
            "TensorlakehouseCollectionRegister._proxy_request",
            new=AsyncMock(return_value=MOCK_STAC_COLLECTIONS),
        ):
            resp = test_client.get("/openeo/1.1.0/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["collections"]) == 1
        assert data["collections"][0]["id"] == "test-collection"

    def test_get_single_collection(self, test_client):
        with patch(
            "openeo_fastapi_app.collections."
            "TensorlakehouseCollectionRegister._proxy_request",
            new=AsyncMock(return_value=MOCK_STAC_COLLECTION),
        ):
            resp = test_client.get("/openeo/1.1.0/collections/test-collection")
        assert resp.status_code == 200
        assert resp.json()["id"] == "test-collection"

    def test_stac_unavailable_returns_503(self, test_client):
        import aiohttp

        async def raise_connection_error(path):
            raise aiohttp.ClientConnectorError(
                connection_key=None,  # type: ignore[arg-type]
                os_error=OSError("Connection refused"),
            )

        with patch(
            "openeo_fastapi_app.collections."
            "TensorlakehouseCollectionRegister._proxy_request",
            new=raise_connection_error,
        ):
            resp = test_client.get("/openeo/1.1.0/collections")
        assert resp.status_code == 503


@requires_openeo_fastapi
class TestJobs:
    def test_list_jobs_empty(self, test_client):
        from openeo_fastapi_app.jobs import _JOB_STORE
        _JOB_STORE.clear()
        resp = test_client.get("/openeo/1.1.0/jobs")
        assert resp.status_code == 200
        assert resp.json()["jobs"] == []

    def test_create_and_get_job(self, test_client):
        from openeo_fastapi_app.jobs import _JOB_STORE
        _JOB_STORE.clear()

        body = {
            "process": {
                "id": "test-process",
                "process_graph": {
                    "lc1": {
                        "process_id": "load_collection",
                        "arguments": {"id": "test-collection"},
                    }
                },
            },
            "title": "My test job",
        }
        create_resp = test_client.post("/openeo/1.1.0/jobs", json=body)
        assert create_resp.status_code == 201
        job_id = create_resp.headers.get("OpenEO-Identifier")
        assert job_id

        get_resp = test_client.get(f"/openeo/1.1.0/jobs/{job_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == job_id
        assert get_resp.json()["status"] == "created"

    def test_get_nonexistent_job_returns_404(self, test_client):
        resp = test_client.get("/openeo/1.1.0/jobs/nonexistent-id")
        assert resp.status_code == 404


@requires_openeo_fastapi
class TestProcesses:
    def test_list_processes(self, test_client):
        resp = test_client.get("/openeo/1.1.0/processes")
        assert resp.status_code == 200
        data = resp.json()
        assert "processes" in data
        assert len(data["processes"]) > 0
