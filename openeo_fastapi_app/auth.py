"""Authentication override for the Tensorlakehouse OpenEO FastAPI backend.

The original openeo-fastapi ``Authenticator.validate`` dependency requires a
live OIDC issuer.  For local / filesystem-only deployments we provide a
pass-through authenticator that creates an anonymous user so the server is
immediately usable without any identity provider.

To re-enable real OIDC authentication set the environment variable
``OPENEO_AUTH_ENABLED=true``.
"""

import os
import uuid
import datetime
import logging

logger = logging.getLogger(__name__)

_ANONYMOUS_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_ANONYMOUS_OIDC_SUB = "anonymous"

AUTH_ENABLED = os.environ.get("OPENEO_AUTH_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)


# ---------------------------------------------------------------------------
# Minimal User-like object that satisfies the openeo-fastapi interface without
# depending on a live database.
# ---------------------------------------------------------------------------

try:
    from openeo_fastapi.client.auth import User as _FastapiUser

    class AnonymousUser(_FastapiUser):
        user_id: uuid.UUID = _ANONYMOUS_USER_ID
        oidc_sub: str = _ANONYMOUS_OIDC_SUB
        created_at: datetime.datetime = datetime.datetime(2024, 1, 1)

except ImportError:
    # Fallback if openeo-fastapi not installed yet
    class AnonymousUser:  # type: ignore[no-redef]
        user_id = _ANONYMOUS_USER_ID
        oidc_sub = _ANONYMOUS_OIDC_SUB


def anonymous_validator(authorization=None):
    """FastAPI dependency that always returns an anonymous user.

    This completely replaces the openeo-fastapi OIDC flow so the server can
    start without any external identity provider.

    The signature intentionally avoids importing ``fastapi.Header`` at module
    level so that this module can be imported on any Python version for testing
    purposes.  When wired into a FastAPI application via
    ``api.override_authentication(anonymous_validator)`` FastAPI will inject
    the ``Authorization`` header value from the request.
    """
    logger.debug("anonymous_validator – returning AnonymousUser (auth disabled)")
    return AnonymousUser()
