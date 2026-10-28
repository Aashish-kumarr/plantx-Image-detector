"""Health check endpoint route definition."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.db.client import is_connected
from app.core.logging import get_logger

logger = get_logger("routes.health")

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    database: str
    storage: str


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Return current truthful backend service status.

    Reports only components that are genuinely initialized.
    Does NOT report ML models, Firebase, CUDA, or cloud storage.
    """
    from app.storage.local import LocalStorage
    import os

    db_status = "connected" if is_connected() else "unavailable"

    try:
        storage_root = settings.LOCAL_STORAGE_ROOT
        storage_status = "ready" if os.path.isdir(storage_root) else "not_initialized"
    except Exception:  # noqa: BLE001
        storage_status = "error"

    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        environment=settings.APP_ENV,
        database=db_status,
        storage=storage_status,
    )
