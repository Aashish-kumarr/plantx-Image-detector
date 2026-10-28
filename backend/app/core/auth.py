"""Firebase Admin Authentication module for PlantDx FastAPI backend."""

import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Request
import firebase_admin
from firebase_admin import auth, credentials

from app.core.config import settings
from app.core.errors import UnauthorizedError
from app.core.logging import get_logger

logger = get_logger("auth")

_firebase_app: Optional[firebase_admin.App] = None


@dataclass
class AuthUser:
    """Authenticated user representation containing Firebase uid and optional email."""

    uid: str
    email: Optional[str] = None


def init_firebase() -> None:
    """Initialize Firebase Admin SDK if credentials or project ID are provided."""
    global _firebase_app
    if _firebase_app is not None or len(firebase_admin._apps) > 0:  # noqa: SLF001
        _firebase_app = firebase_admin.get_app()
        return

    try:
        if settings.FIREBASE_CREDENTIALS_PATH and os.path.exists(settings.FIREBASE_CREDENTIALS_PATH):
            cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
            _firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized with certificate file.")
        elif settings.FIREBASE_PROJECT_ID:
            _firebase_app = firebase_admin.initialize_app(
                options={"projectId": settings.FIREBASE_PROJECT_ID}
            )
            logger.info("Firebase Admin initialized with project ID: %s", settings.FIREBASE_PROJECT_ID)
        else:
            try:
                _firebase_app = firebase_admin.initialize_app()
                logger.info("Firebase Admin initialized with default credentials.")
            except Exception as e:  # noqa: BLE001
                logger.debug("Firebase Admin default initialization skipped: %s", e)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to initialize Firebase Admin SDK: %s", exc)


def verify_id_token(token: str) -> AuthUser:
    """Verify a Firebase ID token using Firebase Admin SDK and return an AuthUser object."""
    init_firebase()

    if len(firebase_admin._apps) == 0:  # noqa: SLF001
        raise UnauthorizedError("Firebase authentication service is not configured on the backend.")

    try:
        decoded_token = auth.verify_id_token(token, check_revoked=True)
        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        if not uid:
            raise UnauthorizedError("Invalid token payload: missing uid.")
        return AuthUser(uid=str(uid), email=str(email) if email else None)
    except auth.InvalidIdTokenError as exc:
        raise UnauthorizedError("Invalid authentication token.") from exc
    except auth.ExpiredIdTokenError as exc:
        raise UnauthorizedError("Authentication token has expired.") from exc
    except Exception as exc:
        logger.warning("Token verification failed: %s", exc)
        raise UnauthorizedError("Authentication token verification failed.") from exc


async def get_current_user(request: Request) -> AuthUser:
    """FastAPI dependency to extract and verify Bearer token from Authorization header.

    Runtime strict authentication:
    - Requires Authorization: Bearer <token> header.
    - Verifies token against Firebase Admin SDK via verify_id_token().
    - Contains NO dev bypasses, synthetic token checks, or fallback users.
    """
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        raise UnauthorizedError("Missing Authorization header.")

    parts = auth_header.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise UnauthorizedError("Invalid Authorization header format. Expected 'Bearer <token>'.")

    token = parts[1]
    return verify_id_token(token)
