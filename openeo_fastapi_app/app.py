"""Main application factory for the Tensorlakehouse OpenEO FastAPI backend.

This module wires together:

* ``OpenEOCore``  – the openeo-fastapi client layer
* ``OpenEOApi``   – the openeo-fastapi FastAPI wrapper
* Custom registers for collections, jobs and processes
* Anonymous authentication override (no OIDC required for local use)

Usage
-----
Start with uvicorn::

    uvicorn openeo_fastapi_app.app:app --host 127.0.0.1 --port 9091

Or via the convenience script at the repo root::

    python run_server.py

Environment variables
---------------------
``STAC_URL``       URL of the STAC API to proxy (default: http://localhost:8080/)
``DATA_ROOT``      Root directory that contains EO data on the local filesystem
``API_DNS``        Public hostname:port for the API (default: localhost:9091)
``API_TLS``        Set to ``true`` to advertise https:// in well-known (default: false)
``OPENEO_AUTH_ENABLED``  Set to ``true`` to enable real OIDC authentication
"""

import logging
import os

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build the FastAPI application
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Application factory.  Returns a configured FastAPI instance."""

    from openeo_fastapi.api.app import OpenEOApi
    from openeo_fastapi.api.types import Billing, FileFormat, GisDataType, Link, Plan
    from openeo_fastapi.client.core import OpenEOCore

    from openeo_fastapi_app.auth import anonymous_validator, AUTH_ENABLED
    from openeo_fastapi_app.collections import TensorlakehouseCollectionRegister
    from openeo_fastapi_app.jobs import TensorlakehouseJobsRegister
    from openeo_fastapi_app.processes import TensorlakehouseProcessRegister
    from openeo_fastapi_app.settings import Settings

    settings = Settings()

    logger.info(
        "Starting Tensorlakehouse OpenEO FastAPI backend\n"
        "  STAC_API_URL : %s\n"
        "  DATA_ROOT    : %s\n"
        "  API_DNS      : %s\n"
        "  AUTH_ENABLED : %s",
        settings.STAC_API_URL,
        settings.DATA_ROOT,
        settings.API_DNS,
        AUTH_ENABLED,
    )

    links = [
        Link(
            href="https://github.com/IBM/tensorlakehouse-openeo-driver",
            rel="about",
            type="text/html",
            title="Tensorlakehouse OpenEO Driver repository",
        )
    ]

    billing = Billing(
        currency="free",
        default_plan="free",
        plans=[Plan(name="free", description="Free plan.", paid=False)],
    )

    input_formats = [
        FileFormat(
            title="GeoJSON",
            gis_data_types=[GisDataType("vector")],
            parameters={},
        ),
        FileFormat(
            title="GPKG",
            gis_data_types=[GisDataType("raster"), GisDataType("vector")],
            parameters={},
        ),
    ]

    output_formats = [
        FileFormat(
            title="GTiff",
            gis_data_types=[GisDataType("raster")],
            parameters={},
        ),
        FileFormat(
            title="netCDF",
            gis_data_types=[GisDataType("raster")],
            parameters={},
        ),
        FileFormat(
            title="ZARR",
            gis_data_types=[GisDataType("raster")],
            parameters={},
        ),
        FileFormat(
            title="PARQUET",
            gis_data_types=[GisDataType("vector")],
            parameters={},
        ),
    ]

    client = OpenEOCore(
        input_formats=input_formats,
        output_formats=output_formats,
        links=links,
        billing=billing,
        settings=settings,
        collections=TensorlakehouseCollectionRegister(settings),
        jobs=TensorlakehouseJobsRegister(settings, links),
        processes=TensorlakehouseProcessRegister(links),
    )

    fastapi_app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.OPENEO_VERSION,
    )

    api = OpenEOApi(client=client, app=fastapi_app)

    if not AUTH_ENABLED:
        logger.info(
            "Authentication is DISABLED – all requests will be served as the "
            "anonymous user.  Set OPENEO_AUTH_ENABLED=true to enable OIDC."
        )
        api.override_authentication(anonymous_validator)
    else:
        logger.info("Authentication is ENABLED (OIDC via %s)", settings.OIDC_URL)

    return fastapi_app


# Module-level app instance consumed by ``uvicorn openeo_fastapi_app.app:app``
app = create_app()
