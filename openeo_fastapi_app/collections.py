"""CollectionRegister subclass for the Tensorlakehouse OpenEO FastAPI backend.

Instead of relying solely on the openeo-fastapi default which proxies an
external STAC API over HTTP, this implementation also supports a *local*
filesystem-based STAC catalogue built by ``stac-fastapi-filesystem``.

The STAC URL is resolved from:
  1. Environment variable ``STAC_URL`` / ``STAC_API_URL``
  2. Fall-back: ``http://localhost:8080/``

All heavy lifting (catalogue querying, filtering, validation) is delegated
to the parent ``CollectionRegister`` which already handles the
aiohttp-proxied STAC calls.  We only override ``get_collections`` to add
a custom whitelist check and to log useful diagnostics.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import aiohttp
    from fastapi import HTTPException
    from pydantic import ValidationError

    from openeo_fastapi.api.models import Collection, Collections
    from openeo_fastapi.api.types import Error
    from openeo_fastapi.client.collections import CollectionRegister

    class TensorlakehouseCollectionRegister(CollectionRegister):
        """Extended CollectionRegister that logs diagnostics and gracefully handles
        STAC endpoints that return non-standard collection shapes.
        """

        def __init__(self, settings) -> None:
            super().__init__(settings)
            logger.info(
                "TensorlakehouseCollectionRegister initialised with STAC_API_URL=%s",
                settings.STAC_API_URL,
            )

        async def _proxy_request(self, path: str):
            """Proxy a GET request to the configured STAC API.

            Overrides the parent to add structured logging and a useful error
            message when the STAC server is unreachable.
            """
            url = self.settings.STAC_API_URL + path
            logger.debug("STAC proxy → GET %s", url)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            resp = await response.json()
                            logger.debug(
                                "STAC proxy ← 200 from %s (keys=%s)",
                                url,
                                list(resp.keys()) if isinstance(resp, dict) else type(resp),
                            )
                            return resp
                        logger.warning(
                            "STAC proxy ← %s from %s", response.status, url
                        )
                        return None
            except aiohttp.ClientConnectorError as exc:
                logger.error(
                    "Cannot connect to STAC API at %s: %s. "
                    "Start a STAC server (e.g. stac-fastapi-filesystem) or set "
                    "STAC_URL to a reachable endpoint.",
                    url,
                    exc,
                )
                raise HTTPException(
                    status_code=503,
                    detail=Error(
                        code="ServiceUnavailable",
                        message=f"STAC catalogue unreachable at {url}: {exc}",
                    ),
                )

        async def get_collections(self):
            """Return all collections available in the STAC catalogue.

            Collections that fail pydantic validation are dropped with a warning
            rather than causing the entire endpoint to fail.
            """
            path = "collections"
            resp = await self._proxy_request(path)

            if not resp:
                raise HTTPException(
                    status_code=404,
                    detail=Error(code="NotFound", message="No collections found."),
                )

            raw_collections = resp.get("collections", [])
            logger.info("STAC returned %d raw collections", len(raw_collections))

            valid: list[Collection] = []
            whitelist: Optional[list] = getattr(
                self.settings, "STAC_COLLECTIONS_WHITELIST", None
            )

            for raw in raw_collections:
                cid = raw.get("id")
                if whitelist and cid not in whitelist:
                    logger.debug("Skipping collection %r (not in whitelist)", cid)
                    continue
                try:
                    valid.append(Collection(**raw))
                except (ValidationError, Exception) as exc:
                    logger.warning(
                        "Dropping collection %r – pydantic validation error: %s",
                        cid,
                        exc,
                    )

            links = resp.get("links", [])
            logger.info("Returning %d valid collections", len(valid))
            return Collections(collections=valid, links=links)

        async def get_collection(self, collection_id: str):
            """Return metadata for a single collection."""
            whitelist: Optional[list] = getattr(
                self.settings, "STAC_COLLECTIONS_WHITELIST", None
            )
            not_found = HTTPException(
                status_code=404,
                detail=Error(
                    code="NotFound",
                    message=f"Collection '{collection_id}' not found.",
                ),
            )

            if whitelist and collection_id not in whitelist:
                raise not_found

            resp = await self._proxy_request(f"collections/{collection_id}")
            if resp:
                try:
                    return Collection(**resp)
                except (ValidationError, Exception) as exc:
                    logger.error(
                        "Collection %r failed validation: %s", collection_id, exc
                    )
            raise not_found

        async def get_collection_items(self, collection_id: str):
            """Return items for a single collection (proxied from STAC)."""
            whitelist: Optional[list] = getattr(
                self.settings, "STAC_COLLECTIONS_WHITELIST", None
            )
            not_found = HTTPException(
                status_code=404,
                detail=Error(
                    code="NotFound",
                    message=f"Collection '{collection_id}' not found.",
                ),
            )

            if whitelist and collection_id not in whitelist:
                raise not_found

            resp = await self._proxy_request(
                f"collections/{collection_id}/items"
            )
            if resp:
                return resp
            raise not_found

        async def get_collection_item(self, collection_id: str, item_id: str):
            """Return a single item from a collection (proxied from STAC)."""
            resp = await self._proxy_request(
                f"collections/{collection_id}/items/{item_id}"
            )
            if resp:
                return resp
            raise HTTPException(
                status_code=404,
                detail=Error(
                    code="NotFound",
                    message=f"Item '{item_id}' not found in collection '{collection_id}'.",
                ),
            )

except ImportError as _import_err:
    logger.warning(
        "openeo-fastapi not installed; TensorlakehouseCollectionRegister unavailable. "
        "Install with: pip install openeo-fastapi==2026.8.3  (%s)",
        _import_err,
    )
    TensorlakehouseCollectionRegister = None  # type: ignore[assignment,misc]
