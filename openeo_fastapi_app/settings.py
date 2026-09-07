"""Application settings for the openeo-fastapi migration of tensorlakehouse-openeo-driver.

All values can be supplied via environment variables.  Sensible defaults allow the
server to start with only ``STAC_API_URL`` set.
"""

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Re-export pydantic BaseSettings so callers can do:
#   from openeo_fastapi_app.settings import Settings
# We import lazily so this module remains importable even before openeo-fastapi
# is installed – useful for tooling that just reads constants.
# ---------------------------------------------------------------------------

try:
    from openeo_fastapi.client.settings import AppSettings as _Base
    from pydantic import validator

    class Settings(_Base):
        """Concrete settings for the Tensorlakehouse OpenEO FastAPI backend.

        Every field can be overridden with the matching environment variable
        (upper-case, same name).
        """

        # Populated from environment; fall-backs make local dev easy
        API_DNS: str = os.environ.get("API_DNS", "localhost:9091")
        API_TLS: bool = os.environ.get("API_TLS", "False").lower() in ("1", "true", "yes")
        API_TITLE: str = os.environ.get(
            "API_TITLE", "Tensorlakehouse OpenEO Backend"
        )
        API_DESCRIPTION: str = os.environ.get(
            "API_DESCRIPTION",
            "Tensorlakehouse OpenEO backend powered by openeo-fastapi (EODC Driver)",
        )
        OPENEO_VERSION: str = "1.1.0"
        STAC_API_URL: str = os.environ.get("STAC_URL", "http://localhost:8080/")
        OIDC_URL: str = os.environ.get(
            "OIDC_URL", "https://accounts.google.com"
        )
        OIDC_ORGANISATION: str = os.environ.get("OIDC_ORGANISATION", "egi")
        OIDC_POLICIES: Optional[list] = None

        # The filesystem root for data discovery (used by the file-based STAC
        # loader when STAC_API_URL points to a local stac-fastapi instance)
        DATA_ROOT: str = os.environ.get("DATA_ROOT", os.path.expanduser("~/data"))

        class Config:
            """Pydantic model class config."""

            env_file = ".env"
            env_file_encoding = "utf-8"

            @classmethod
            def parse_env_var(cls, field_name: str, raw_val: str):
                if field_name == "STAC_COLLECTIONS_WHITELIST":
                    return [x.strip() for x in raw_val.split(",") if x.strip()]
                elif field_name == "OIDC_POLICIES":
                    return [x.strip() for x in raw_val.split("&&") if x.strip()]
                return cls.json_loads(raw_val)

except ImportError:
    # Fallback for environments where openeo-fastapi is not yet installed
    class Settings:  # type: ignore[no-redef]
        API_DNS = os.environ.get("API_DNS", "localhost:9091")
        API_TLS = False
        API_TITLE = "Tensorlakehouse OpenEO Backend"
        API_DESCRIPTION = (
            "Tensorlakehouse OpenEO backend powered by openeo-fastapi (EODC Driver)"
        )
        OPENEO_VERSION = "1.1.0"
        OPENEO_PREFIX = f"/openeo/{OPENEO_VERSION}"
        STAC_API_URL = os.environ.get("STAC_URL", "http://localhost:8080/")
        OIDC_URL = os.environ.get("OIDC_URL", "https://accounts.google.com")
        OIDC_ORGANISATION = os.environ.get("OIDC_ORGANISATION", "egi")
        OIDC_POLICIES = None
        DATA_ROOT = os.environ.get("DATA_ROOT", os.path.expanduser("~/data"))
        STAC_COLLECTIONS_WHITELIST = None
        STAC_VERSION = "1.0.0"
