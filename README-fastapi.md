# OpenEO FastAPI Migration Guide

This document describes the migration of `tensorlakehouse-openeo-driver` from
the legacy **openeo-python-driver** (Flask + Celery + Redis) to
**[openeo-fastapi](https://github.com/eodcgmbh/openeo-fastapi)** (the EODC
Driver, FastAPI + uvicorn).

---

## What changed

| Concern | Before | After |
|---|---|---|
| HTTP framework | Flask (WSGI) | FastAPI (ASGI) |
| Process execution queue | Celery + Redis | Thread pool (`concurrent.futures`) |
| Workflow manager | Airflow / Kubeflow | None |
| STAC catalogue | Remote IBM GeoDN.Discovery | Any STAC API (local or remote) |
| Authentication | IBM AppID (OIDC) | Configurable; anonymous by default |
| Batch job state | Celery task registry | In-memory dict (no DB required) |
| Entry point | `gunicorn` / Spark submit | `uvicorn` |

---

## Quick start

### Prerequisites

- **Python 3.10 or 3.11** (openeo-fastapi requires `>=3.10,<3.12`)
- A STAC API endpoint that exposes EO collections and items backed by files on
  the local filesystem.  Options:
  - [stac-fastapi-filesystem](https://github.com/developmentseed/stac-fastapi-filesystem)
  - [stac-fastapi-pgstac](https://github.com/stac-utils/stac-fastapi-pgstac) with
    [pgstac](https://github.com/stac-utils/pgstac)
  - Any public STAC API (e.g. `https://planetarycomputer.microsoft.com/api/stac/v1`)

### Install

```bash
# Create a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements-fastapi.txt
```

### Start a local STAC server (optional)

If you have EO files on disk, use **stac-fastapi-filesystem** to expose them:

```bash
pip install stac-fastapi-filesystem
stac-fastapi-filesystem --data-dir /path/to/your/data --port 8080 &
```

The server will scan for STAC items (`.json` sidecar files or `stac.json`
catalogues) under `--data-dir` and expose them at `http://localhost:8080/`.

### Start the OpenEO server

```bash
# Point the server at your STAC API
export STAC_URL=http://localhost:8080/

# Optionally set the public hostname (for well-known discovery)
export API_DNS=localhost:9091

python run_server.py
```

The OpenEO API is now available at **http://localhost:9091/openeo/1.1.0/**.

---

## Configuration reference

All settings are read from environment variables (or a `.env` file).

| Variable | Default | Description |
|---|---|---|
| `STAC_URL` | `http://localhost:8080/` | STAC API base URL |
| `DATA_ROOT` | `~/data` | Local directory containing EO data files |
| `API_DNS` | `localhost:9091` | Public hostname:port |
| `API_TLS` | `false` | Advertise `https://` in well-known |
| `API_TITLE` | `Tensorlakehouse OpenEO Backend` | Title shown in capabilities |
| `OIDC_URL` | `https://accounts.google.com` | OIDC issuer URL |
| `OIDC_ORGANISATION` | `egi` | OIDC provider short name |
| `OPENEO_AUTH_ENABLED` | `false` | Set `true` to enforce OIDC auth |
| `HOST` | `127.0.0.1` | Bind address for uvicorn |
| `PORT` | `9091` | Bind port for uvicorn |
| `LOG_LEVEL` | `info` | uvicorn log level |
| `RELOAD` | `false` | Enable uvicorn auto-reload (dev only) |

---

## Architecture

```
run_server.py
    └── uvicorn → openeo_fastapi_app.app:app
                        │
                        ├── OpenEOApi (openeo-fastapi)
                        │       └── OpenEOCore
                        │               ├── TensorlakehouseCollectionRegister
                        │               │       └── proxies → STAC API (any)
                        │               ├── TensorlakehouseJobsRegister
                        │               │       └── in-memory store + ThreadPoolExecutor
                        │               └── TensorlakehouseProcessRegister
                        │                       └── openeo-processes-dask-slim specs
                        └── Auth override (anonymous by default)
```

---

## Development

```bash
# Install dev dependencies
pip install -r requirements-fastapi.txt pytest httpx

# Run the tests
pytest openeo_fastapi_app/tests/ -v
```

---

## Key differences from the legacy driver

### No Celery / Redis

Batch jobs now run in a background thread pool (`ThreadPoolExecutor`).  For
production use you can replace `TensorlakehouseJobsRegister` with a
PostgreSQL-backed implementation using the `openeo-fastapi` PSQL helpers.

### No Airflow / Kubeflow

The pipeline handlers in `tensorlakehouse_openeo_driver/pipeline/` are not
used in this migration.  All processing goes through `openeo-processes-dask`.

### STAC-first data discovery

Data discovery is fully delegated to the STAC API.  The server does not need
direct access to COS / S3 / IBM Cloud Object Storage at startup – only the
STAC catalogue URL must be reachable.

### Re-usable process graph execution

`TensorlakehouseProcessing` from the legacy driver is still used inside
`TensorlakehouseJobsRegister._run_process_graph()` so that all existing
process implementations (load_collection, save_result, etc.) continue to work.
