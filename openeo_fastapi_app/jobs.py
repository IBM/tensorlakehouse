"""In-memory batch-jobs register for the Tensorlakehouse OpenEO FastAPI backend.

The original backend used Celery + Redis as its job queue.  This replacement
implementation stores job state entirely in memory using a plain Python dict.
This means:

* No Redis/Celery required.
* No Airflow or any other workload manager required.
* Jobs survive until the server process restarts (suitable for demos and
  development; for production persistence, swap ``_JOB_STORE`` for a
  SQLite / PostgreSQL-backed solution).

Job execution is performed synchronously in a background thread via
``concurrent.futures.ThreadPoolExecutor`` so the HTTP server stays responsive
while the process graph runs.
"""

import datetime
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global in-memory store
# Key: job_id (str)  Value: dict with job metadata
# ---------------------------------------------------------------------------
_JOB_STORE: Dict[str, dict] = {}
_JOB_FUTURES: Dict[str, Future] = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tlh-job")
_STORE_LOCK = threading.Lock()


def _make_job_record(
    job_id: str,
    user_id: str,
    process: dict,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "job_id": job_id,
        "user_id": user_id,
        "process": process,
        "status": "created",
        "created": now,
        "updated": now,
        "title": title,
        "description": description,
        "logs": [],
        "results": None,
    }


def _run_process_graph(job_id: str, process: dict) -> None:
    """Execute the process graph for ``job_id`` in a background thread.

    On success the job status is set to ``"finished"`` and a minimal result
    asset is stored.  On failure the status is set to ``"error"``.
    """
    with _STORE_LOCK:
        if job_id not in _JOB_STORE:
            logger.error("_run_process_graph: job %s not found in store", job_id)
            return
        _JOB_STORE[job_id]["status"] = "running"
        _JOB_STORE[job_id]["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        logger.info("Job %s: starting process graph execution", job_id)

        # Import lazily so the module is importable even when these heavy
        # dependencies are not installed.
        from openeo_pg_parser_networkx import OpenEOProcessGraph
        from tensorlakehouse_openeo_driver.processing import TensorlakehouseProcessing

        processing = TensorlakehouseProcessing()
        pg = process.get("process_graph", process)
        parsed = OpenEOProcessGraph(pg_data={"process_graph": pg} if "process_graph" not in process else process)
        callable_pg = parsed.to_callable(process_registry=processing.process_registry)
        result = callable_pg()

        with _STORE_LOCK:
            _JOB_STORE[job_id]["status"] = "finished"
            _JOB_STORE[job_id]["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            _JOB_STORE[job_id]["results"] = {
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": job_id,
                "geometry": None,
                "properties": {
                    "datetime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "openeo:status": "finished",
                },
                "assets": {},
                "links": [],
            }
            _JOB_STORE[job_id]["logs"].append(
                {
                    "id": str(uuid.uuid4()),
                    "level": "info",
                    "message": "Job completed successfully.",
                    "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
            )
        logger.info("Job %s: finished successfully", job_id)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s: execution failed: %s", job_id, exc)
        with _STORE_LOCK:
            _JOB_STORE[job_id]["status"] = "error"
            _JOB_STORE[job_id]["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            _JOB_STORE[job_id]["logs"].append(
                {
                    "id": str(uuid.uuid4()),
                    "level": "error",
                    "message": str(exc),
                    "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
            )


# ---------------------------------------------------------------------------
# JobsRegister
# ---------------------------------------------------------------------------

try:
    from fastapi import Depends, HTTPException, Response
    from openeo_fastapi.api.models import BatchJob, JobsGetResponse, JobsRequest
    from openeo_fastapi.api.types import Endpoint, Error
    from openeo_fastapi.client.auth import Authenticator, User
    from openeo_fastapi.client.register import EndpointRegister

    JOBS_ENDPOINTS = [
        Endpoint(path="/jobs", methods=["GET"]),
        Endpoint(path="/jobs", methods=["POST"]),
        Endpoint(path="/jobs/{job_id}", methods=["GET"]),
        Endpoint(path="/jobs/{job_id}", methods=["PATCH"]),
        Endpoint(path="/jobs/{job_id}", methods=["DELETE"]),
        Endpoint(path="/jobs/{job_id}/estimate", methods=["GET"]),
        Endpoint(path="/jobs/{job_id}/logs", methods=["GET"]),
        Endpoint(path="/jobs/{job_id}/results", methods=["GET"]),
        Endpoint(path="/jobs/{job_id}/results", methods=["POST"]),
        Endpoint(path="/jobs/{job_id}/results", methods=["DELETE"]),
    ]

    class TensorlakehouseJobsRegister(EndpointRegister):
        """In-memory JobsRegister – no Celery/Redis/Airflow required.

        Job state is kept in the module-level ``_JOB_STORE`` dict and execution
        is dispatched to a ``ThreadPoolExecutor``.
        """

        def __init__(self, settings, links) -> None:
            super().__init__()
            self.endpoints = JOBS_ENDPOINTS
            self.settings = settings
            self.links = links

        # ------------------------------------------------------------------
        # Helper
        # ------------------------------------------------------------------

        @staticmethod
        def _get_or_404(job_id: str) -> dict:
            job = _JOB_STORE.get(str(job_id))
            if job is None:
                raise HTTPException(
                    status_code=404,
                    detail=Error(
                        code="JobNotFound",
                        message=f"No job found with id: {job_id}",
                    ),
                )
            return job

        @staticmethod
        def _to_batch_job(record: dict) -> BatchJob:
            return BatchJob(
                id=record["job_id"],
                status=record["status"],
                created=record["created"],
                title=record.get("title"),
                description=record.get("description"),
                process=record.get("process"),
            )

        # ------------------------------------------------------------------
        # Endpoints
        # ------------------------------------------------------------------

        def list_jobs(
            self,
            limit: Optional[int] = 10,
            user: User = Depends(Authenticator.validate),
        ) -> JobsGetResponse:
            uid = str(user.user_id)
            with _STORE_LOCK:
                jobs = [
                    self._to_batch_job(r)
                    for r in _JOB_STORE.values()
                    if r["user_id"] == uid
                ]
            return JobsGetResponse(jobs=jobs[:limit], links=[])

        def create_job(
            self,
            body: JobsRequest,
            user: User = Depends(Authenticator.validate),
        ) -> Response:
            job_id = str(uuid.uuid4())
            process_dict = (
                body.process.dict() if hasattr(body.process, "dict") else {}
            )
            record = _make_job_record(
                job_id=job_id,
                user_id=str(user.user_id),
                process=process_dict,
                title=body.title,
                description=body.description,
            )
            with _STORE_LOCK:
                _JOB_STORE[job_id] = record
            logger.info("Job %s created for user %s", job_id, user.user_id)
            return Response(
                status_code=201,
                headers={
                    "Location": f"{self.settings.API_DNS}"
                    f"{self.settings.OPENEO_PREFIX}/jobs/{job_id}",
                    "OpenEO-Identifier": job_id,
                    "access-control-expose-headers": (
                        "Location, OpenEO-Identifier"
                    ),
                },
            )

        def update_job(
            self,
            job_id: uuid.UUID,
            body: JobsRequest,
            user: User = Depends(Authenticator.validate),
        ) -> Response:
            record = self._get_or_404(str(job_id))
            if body.title is not None:
                record["title"] = body.title
            if body.description is not None:
                record["description"] = body.description
            if body.process is not None:
                record["process"] = (
                    body.process.dict()
                    if hasattr(body.process, "dict")
                    else body.process
                )
            record["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return Response(status_code=204)

        def get_job(
            self,
            job_id: uuid.UUID,
            user: User = Depends(Authenticator.validate),
        ) -> BatchJob:
            return self._to_batch_job(self._get_or_404(str(job_id)))

        def delete_job(
            self,
            job_id: uuid.UUID,
            user: User = Depends(Authenticator.validate),
        ) -> Response:
            self._get_or_404(str(job_id))
            with _STORE_LOCK:
                _JOB_STORE.pop(str(job_id), None)
                future = _JOB_FUTURES.pop(str(job_id), None)
                if future and not future.done():
                    future.cancel()
            return Response(status_code=204)

        def estimate(
            self,
            job_id: uuid.UUID,
            user: User = Depends(Authenticator.validate),
        ):
            raise HTTPException(
                status_code=501,
                detail=Error(
                    code="FeatureUnsupported",
                    message="Cost estimation not supported in this deployment.",
                ),
            )

        def logs(
            self,
            job_id: uuid.UUID,
            user: User = Depends(Authenticator.validate),
        ):
            record = self._get_or_404(str(job_id))
            from openeo_fastapi.api.models import JobsGetLogsResponse

            log_entries = record.get("logs", [])
            return JobsGetLogsResponse(logs=log_entries, links=[])

        def get_results(
            self,
            job_id: uuid.UUID,
            user: User = Depends(Authenticator.validate),
        ):
            record = self._get_or_404(str(job_id))
            if record["status"] != "finished":
                raise HTTPException(
                    status_code=400,
                    detail=Error(
                        code="JobNotFinished",
                        message=f"Job {job_id} has not finished (status: {record['status']}).",
                    ),
                )
            return record["results"]

        def start_job(
            self,
            job_id: uuid.UUID,
            user: User = Depends(Authenticator.validate),
        ) -> Response:
            record = self._get_or_404(str(job_id))
            if record["status"] in ("running", "finished"):
                raise HTTPException(
                    status_code=400,
                    detail=Error(
                        code="JobLocked",
                        message=f"Job {job_id} is already {record['status']}.",
                    ),
                )
            # Submit to thread pool
            process = record["process"]
            future = _EXECUTOR.submit(_run_process_graph, str(job_id), process)
            with _STORE_LOCK:
                _JOB_FUTURES[str(job_id)] = future
            logger.info("Job %s submitted for execution", job_id)
            return Response(status_code=202)

        def cancel_job(
            self,
            job_id: uuid.UUID,
            user: User = Depends(Authenticator.validate),
        ) -> Response:
            record = self._get_or_404(str(job_id))
            future = _JOB_FUTURES.get(str(job_id))
            if future and not future.done():
                future.cancel()
            record["status"] = "canceled"
            record["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return Response(status_code=204)

        def process_sync_job(
            self,
            body: JobsRequest,
            user: User = Depends(Authenticator.validate),
        ):
            """Execute a synchronous (non-batch) job and return the result directly."""
            raise HTTPException(
                status_code=501,
                detail=Error(
                    code="FeatureUnsupported",
                    message=(
                        "Synchronous processing is not yet supported in this deployment. "
                        "Create a batch job with POST /jobs and start it with POST /jobs/{job_id}/results."
                    ),
                ),
            )

except ImportError as _import_err:
    logger.warning(
        "openeo-fastapi not installed; TensorlakehouseJobsRegister unavailable (%s)",
        _import_err,
    )
    TensorlakehouseJobsRegister = None  # type: ignore[assignment,misc]
