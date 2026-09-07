"""Structural tests – importable on any Python version including 3.14.

These tests verify module layout and fallback behaviour when openeo-fastapi
is not installed.
"""

import importlib
import sys

import pytest


class TestModuleImports:
    """Verify all migration modules import cleanly."""

    def test_import_auth(self):
        mod = importlib.import_module("openeo_fastapi_app.auth")
        assert hasattr(mod, "anonymous_validator")
        assert hasattr(mod, "AUTH_ENABLED")

    def test_import_settings(self):
        mod = importlib.import_module("openeo_fastapi_app.settings")
        assert hasattr(mod, "Settings")

    def test_import_collections(self):
        mod = importlib.import_module("openeo_fastapi_app.collections")
        # TensorlakehouseCollectionRegister is None when openeo-fastapi absent
        assert hasattr(mod, "TensorlakehouseCollectionRegister")

    def test_import_jobs(self):
        mod = importlib.import_module("openeo_fastapi_app.jobs")
        assert hasattr(mod, "TensorlakehouseJobsRegister")
        # In-memory store is always present
        assert hasattr(mod, "_JOB_STORE")

    def test_import_processes(self):
        mod = importlib.import_module("openeo_fastapi_app.processes")
        assert hasattr(mod, "TensorlakehouseProcessRegister")


class TestAuth:
    """Test the anonymous authenticator."""

    def test_auth_disabled_by_default(self):
        import os
        os.environ["OPENEO_AUTH_ENABLED"] = "false"
        # Re-import to pick up env change
        import importlib
        mod = importlib.import_module("openeo_fastapi_app.auth")
        importlib.reload(mod)
        assert mod.AUTH_ENABLED is False

    def test_anonymous_validator_returns_user(self):
        """The anonymous validator must return an object with a user_id attr."""
        from openeo_fastapi_app.auth import anonymous_validator
        user = anonymous_validator(authorization=None)
        assert hasattr(user, "user_id")

    def test_anonymous_user_id_is_stable(self):
        """The anonymous user always gets the same UUID."""
        import uuid
        from openeo_fastapi_app.auth import anonymous_validator, _ANONYMOUS_USER_ID
        user = anonymous_validator(authorization=None)
        assert str(user.user_id) == str(_ANONYMOUS_USER_ID)


class TestSettings:
    """Test settings defaults."""

    def test_settings_instantiation(self):
        """Settings must be instantiable with only env vars (no .env file)."""
        import importlib
        mod = importlib.import_module("openeo_fastapi_app.settings")
        settings = mod.Settings()
        assert settings.API_DNS == "localhost:9091"
        assert "localhost:8080" in settings.STAC_API_URL or "example.com" in settings.STAC_API_URL

    def test_stac_url_from_env(self, monkeypatch):
        monkeypatch.setenv("STAC_URL", "http://my-stac.example.com/")
        import importlib
        mod = importlib.import_module("openeo_fastapi_app.settings")
        importlib.reload(mod)
        settings = mod.Settings()
        assert "my-stac.example.com" in settings.STAC_API_URL


class TestJobStore:
    """Test in-memory job store operations (no openeo-fastapi required)."""

    def setup_method(self):
        from openeo_fastapi_app.jobs import _JOB_STORE
        _JOB_STORE.clear()

    def test_make_job_record(self):
        from openeo_fastapi_app.jobs import _make_job_record
        rec = _make_job_record(
            job_id="abc-123",
            user_id="user-1",
            process={"process_graph": {}},
            title="My job",
        )
        assert rec["job_id"] == "abc-123"
        assert rec["status"] == "created"
        assert rec["title"] == "My job"

    def test_job_store_is_dict(self):
        from openeo_fastapi_app.jobs import _JOB_STORE, _make_job_record
        _JOB_STORE.clear()
        rec = _make_job_record("j1", "u1", {})
        _JOB_STORE["j1"] = rec
        assert "j1" in _JOB_STORE
        assert _JOB_STORE["j1"]["status"] == "created"

    def test_job_store_status_update(self):
        from openeo_fastapi_app.jobs import _JOB_STORE, _make_job_record
        _JOB_STORE.clear()
        _JOB_STORE["j2"] = _make_job_record("j2", "u1", {})
        _JOB_STORE["j2"]["status"] = "running"
        assert _JOB_STORE["j2"]["status"] == "running"
