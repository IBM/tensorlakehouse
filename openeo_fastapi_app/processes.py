"""ProcessRegister for the Tensorlakehouse OpenEO FastAPI backend.

Extends the default openeo-fastapi ``ProcessRegister`` to use the full
``openeo-processes-dask-slim`` specification catalogue.  No database required.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from openeo_fastapi.client.processes import ProcessRegister

    class TensorlakehouseProcessRegister(ProcessRegister):
        """Process register that uses openeo-processes-dask-slim specs.

        No custom processes are added on top of the predefined ones; this can
        be extended in future by registering additional specs in
        ``_create_process_registry``.
        """

        def __init__(self, links) -> None:
            super().__init__(links)
            logger.info(
                "TensorlakehouseProcessRegister initialised with %d predefined processes",
                len(list(self.process_registry["predefined", None].values())),
            )

except ImportError as _import_err:
    logger.warning(
        "openeo-fastapi not installed; TensorlakehouseProcessRegister unavailable (%s)",
        _import_err,
    )
    TensorlakehouseProcessRegister = None  # type: ignore[assignment,misc]
