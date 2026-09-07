"""Test fixtures for the openeo-fastapi migration tests."""

import os
import sys

import pytest

# Provide minimal env vars so Settings() can be instantiated in tests
# without a .env file.
os.environ.setdefault("STAC_URL", "http://test-stac.example.com/")
os.environ.setdefault("API_DNS", "localhost:9091")
os.environ.setdefault("API_TITLE", "Test Tensorlakehouse OpenEO")
os.environ.setdefault("API_DESCRIPTION", "Test instance")
os.environ.setdefault("OIDC_URL", "https://accounts.example.com")
os.environ.setdefault("OIDC_ORGANISATION", "test")
os.environ.setdefault("OPENEO_AUTH_ENABLED", "false")

# Determine if openeo-fastapi is available (requires Python <3.12)
try:
    import openeo_fastapi  # noqa: F401
    OPENEO_FASTAPI_AVAILABLE = True
except ImportError:
    OPENEO_FASTAPI_AVAILABLE = False

requires_openeo_fastapi = pytest.mark.skipif(
    not OPENEO_FASTAPI_AVAILABLE,
    reason="openeo-fastapi not installed (requires Python >=3.10,<3.12)",
)
