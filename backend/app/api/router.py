"""Centralized API v1 router definition."""

from fastapi import APIRouter
from app.api.routes import health
from app.api.routes import images
from app.api.routes import quality
from app.api.routes import analyses

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(images.router)
api_router.include_router(quality.router)
api_router.include_router(analyses.router)
