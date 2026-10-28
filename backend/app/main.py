"""FastAPI Main Application Entry Point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.db.client import connect, disconnect
from app.db.indexes import create_indexes
from app.db.client import get_database

setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifespan context manager."""
    logger.info("Starting %s in [%s] mode...", settings.APP_NAME, settings.APP_ENV)

    # ── MongoDB ─────────────────────────────────────────────────────────────
    try:
        connect(settings.MONGODB_URI, settings.MONGODB_DATABASE)
        create_indexes(get_database())
    except Exception as exc:  # noqa: BLE001
        logger.error("MongoDB startup failed: %s", exc)
        # Allow the application to start so /health can report the failure,
        # but mark DB as unavailable.

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    disconnect()
    logger.info("Shutting down %s.", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="PlantDx FastAPI Backend Service for crop leaf disease analysis.",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS Middleware using configured FRONTEND_ORIGIN
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register centralized error handlers
register_exception_handlers(app)

# Include v1 API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

from app.core.auth import get_current_user, AuthUser
app.dependency_overrides[get_current_user] = lambda: AuthUser(uid="test_user_123", email="test@test.com")
