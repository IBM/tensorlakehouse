#!/usr/bin/env python
"""Convenience entry point to launch the Tensorlakehouse OpenEO FastAPI server.

Starts a uvicorn ASGI server on 127.0.0.1:9091 (or whatever is configured via
environment variables).  No Airflow, no Celery, no Redis required.

Usage::

    python run_server.py

Environment variables
---------------------
``HOST``                 Bind address (default: 127.0.0.1)
``PORT``                 Bind port    (default: 9091)
``LOG_LEVEL``            uvicorn log level (default: info)
``STAC_URL``             STAC API base URL (default: http://localhost:8080/)
``DATA_ROOT``            Local directory containing EO data files
``OPENEO_AUTH_ENABLED``  Set to ``true`` to enable OIDC authentication

Quick-start with a local stac-fastapi-filesystem backend
--------------------------------------------------------
1. Install stac-fastapi-filesystem and point it at your data directory::

       pip install stac-fastapi-filesystem
       stac-fastapi-filesystem --data-dir /path/to/data --port 8080 &

2. Start this server::

       STAC_URL=http://localhost:8080/ python run_server.py

3. Explore the OpenEO API at http://localhost:9091/openeo/1.1.0/
"""

import logging
import os
import sys

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("tlh.server")


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "9091"))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    reload = os.environ.get("RELOAD", "false").lower() in ("1", "true", "yes")

    logger.info("Tensorlakehouse OpenEO FastAPI server")
    logger.info("  Binding  : %s:%s", host, port)
    logger.info("  STAC URL : %s", os.environ.get("STAC_URL", "http://localhost:8080/"))
    logger.info("  Data root: %s", os.environ.get("DATA_ROOT", "~/data"))

    uvicorn.run(
        "openeo_fastapi_app.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


if __name__ == "__main__":
    main()
