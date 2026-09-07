"""Tests for the openeo-fastapi migration of tensorlakehouse-openeo-driver.

These tests can run in two modes:

1. **Without openeo-fastapi installed** – structural tests that exercise the
   module imports and validate that the fallback code paths work correctly.

2. **With openeo-fastapi installed** (Python 3.10/3.11 env) – full integration
   tests that start a TestClient and exercise all endpoints.

Run with::

    pytest openeo_fastapi_app/tests/ -v
"""
